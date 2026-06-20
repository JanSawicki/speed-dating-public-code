import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import List, Type

from cache import DiskCache

_TYPE_MAP = {
    "HumanMessage": HumanMessage,
    "SystemMessage": SystemMessage,
    "AIMessage": AIMessage,
}


def _serialize_messages(messages: List) -> str:
    return json.dumps([{"type": type(m).__name__, "content": m.content} for m in messages])


def _deserialize_messages(messages_json: str) -> List:
    return [_TYPE_MAP[d["type"]](content=d["content"]) for d in json.loads(messages_json)]


class CachedModel:
    """Wraps a LangChain chat model, caching all invoke calls to disk via @cache."""

    def __init__(self, model, cache_dir: str, cache_batch_size: int = 20, verbose: bool = False):
        self._model = model
        self._structured_fns: dict = {}
        self._disk_cache = DiskCache(cache_dir=cache_dir, batch_size=cache_batch_size, verbose=verbose)

        model_ref = model

        def _invoke(messages_json: str) -> str:
            return model_ref.invoke(_deserialize_messages(messages_json)).content

        self._invoke_fn = self._disk_cache(_invoke)

    def invoke(self, messages: List) -> AIMessage:
        content = self._invoke_fn(_serialize_messages(messages))
        return AIMessage(content=content)

    def with_structured_output(self, schema: Type[BaseModel]):
        name = schema.__name__
        if name not in self._structured_fns:
            model_ref = self._model

            def _structured(messages_json: str) -> str:
                messages = _deserialize_messages(messages_json)
                result = model_ref.with_structured_output(schema).invoke(messages)
                if result is not None:
                    return result.model_dump_json()
                # Fallback: ask the model to produce JSON directly
                schema_str = json.dumps(schema.model_json_schema(), indent=2)
                augmented = messages + [HumanMessage(
                    content=f"Respond with a JSON object matching this schema exactly:\n{schema_str}\nOutput only valid JSON, no other text."
                )]
                content = model_ref.bind(max_tokens=4096).invoke(augmented).content.strip()
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.split("```")[0].strip()
                return schema.model_validate_json(content).model_dump_json()

            _structured.__name__ = f"_structured_{name}"
            self._structured_fns[name] = self._disk_cache(_structured)

        fn = self._structured_fns[name]

        class _StructuredWrapper:
            def invoke(self, messages: List) -> BaseModel:
                return schema.model_validate_json(fn(_serialize_messages(messages)))

        return _StructuredWrapper()

    def save_caches(self) -> None:
        self._invoke_fn.save()
        for fn in self._structured_fns.values():
            fn.save()
