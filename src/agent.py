from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class Match(BaseModel):
    agent_id: str = Field(description="The exact conversation label from the header, e.g. 'SUBJECT_L0_INCOMPATIBLE__run0'")
    compatibility_score: int = Field(description="Compatibility score 0–100. Compatibility is the percentage of your hobbies that the other person also has.")
    explanation: str = Field(description="Explanation of the score, referencing the specific hobbies discussed")



class Agent:
    def __init__(
        self,
        id: str,
        hobbies: List[str],
        system_prompt: str,
        model,
        get_matches_prompt: str,
        scoring_model=None,
        logging: bool = False,
        name: str = "",
    ):
        self.id = id
        self.name = name
        self.hobbies = hobbies
        self.log_name = f"{self.id}_{self.name}" if self.name else self.id
        self.system_prompt = system_prompt
        self.model = model
        self.scoring_model = scoring_model if scoring_model is not None else model
        self.get_matches_prompt = get_matches_prompt
        self.conversations: Dict[str, List] = {}
        self.logging = logging

    def _conv_key(self, other: "Agent", run_id: int) -> str:
        return f"{other.id}__run{run_id}"

    def initial_message(self, other: "Agent", system_message: str, run_id: int = 0) -> str:
        key = self._conv_key(other, run_id)
        self.conversations[key] = [
            SystemMessage(content=f"[Run: {run_id}]"),
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=system_message),
        ]
        message = self.model.invoke(self.conversations[key])
        self.conversations[key].append(message)
        if self.logging:
            print(f"{self.log_name} to {other.log_name}: {message.content}")
        return message.content

    def respond(self, other: "Agent", content: str, run_id: int = 0) -> str:
        key = self._conv_key(other, run_id)
        if key not in self.conversations:
            self.conversations[key] = [
                SystemMessage(content=f"[Run: {run_id}]"),
                SystemMessage(content=self.system_prompt),
            ]
        history = self.conversations[key]
        history.append(HumanMessage(content=content))
        response = self.model.invoke(history)
        history.append(response)
        if self.logging:
            print(f"{self.log_name} to {other.log_name}: {response.content}")
        return response.content

    def get_matches(self, run_id: int, bar=None) -> List[Match]:
        results = []
        for key, msgs in self.conversations.items():
            if not key.endswith(f"__run{run_id}"):
                continue
            conversation = (
                f"\n\n\n---------- Conversation with {key}: ----------\n\n\n"
                + "\n".join(m.content for m in msgs if not isinstance(m, SystemMessage))
            )
            messages = [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=self.get_matches_prompt),
                HumanMessage(content=conversation),
            ]
            result: Match = self.scoring_model.with_structured_output(Match).invoke(messages)
            results.append(result)
            if bar is not None:
                bar.update(1)
        self.scoring_model.save_caches()
        return results
