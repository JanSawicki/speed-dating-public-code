import os
import json
from functools import wraps
from typing import Any, Callable


class DiskCache:
    def __init__(self, cache_dir: str = "cache_data", batch_size: int = 20, verbose: bool = False) -> None:
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.verbose = verbose
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _key_to_str(key: tuple) -> str:
        args, kwargs_items = key
        return json.dumps([list(args), [list(item) for item in kwargs_items]])

    @staticmethod
    def _key_from_str(s: str) -> tuple:
        parts = json.loads(s)
        return (tuple(parts[0]), tuple(tuple(item) for item in parts[1]))

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        file_path = os.path.join(self.cache_dir, f"{func.__name__}.json")

        _mem_cache: dict[Any, Any] = {}
        _new_entry_count = [0]

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                for k_str, v in data.items():
                    _mem_cache[self._key_from_str(k_str)] = v
                if self.verbose:
                    print(f"[cache] '{func.__name__}': loaded {len(_mem_cache)} entries from {file_path}")
            except Exception as e:
                print(f"[cache] '{func.__name__}': failed to load from {file_path}: {e}")

        def save_to_disk() -> None:
            new = _new_entry_count[0]
            serialized = {self._key_to_str(k): v for k, v in _mem_cache.items()}
            with open(file_path, "w") as f:
                json.dump(serialized, f, indent=2)
            _new_entry_count[0] = 0
            if self.verbose:
                print(f"[cache] '{func.__name__}': saved {new} new entries ({len(_mem_cache)} total) to {file_path}")

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            if key in _mem_cache:
                return _mem_cache[key]
            result = func(*args, **kwargs)
            _mem_cache[key] = result
            _new_entry_count[0] += 1
            if _new_entry_count[0] >= self.batch_size:
                save_to_disk()
            return result

        wrapper.save = save_to_disk
        return wrapper


