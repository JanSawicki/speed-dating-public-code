import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import Agent, Match


def _serialize_message(msg) -> dict:
    if isinstance(msg, SystemMessage):
        return {"type": "system", "content": msg.content}
    if isinstance(msg, HumanMessage):
        return {"type": "human", "content": msg.content}
    return {"type": "ai", "content": msg.content}


def _deserialize_message(d: dict):
    t, c = d["type"], d["content"]
    if t == "system":
        return SystemMessage(content=c)
    if t == "human":
        return HumanMessage(content=c)
    return AIMessage(content=c)


def _serialize_agent(agent: Agent) -> dict:
    return {
        "id": agent.id,
        "name": agent.name,
        "hobbies": agent.hobbies,
        "system_prompt": agent.system_prompt,
        "get_matches_prompt": agent.get_matches_prompt,
        "conversations": {
            k: [_serialize_message(m) for m in msgs]
            for k, msgs in agent.conversations.items()
        },
    }


def _deserialize_agent(d: dict, model, scoring_model=None) -> Agent:
    agent = Agent(
        id=d["id"],
        name=d["name"],
        hobbies=d["hobbies"],
        system_prompt=d["system_prompt"],
        model=model,
        scoring_model=scoring_model,
        get_matches_prompt=d["get_matches_prompt"],
    )
    agent.conversations = {
        k: [_deserialize_message(m) for m in msgs]
        for k, msgs in d["conversations"].items()
    }
    return agent


def save_simulation_state(results: Dict[str, tuple], data_path: str) -> None:
    state_dir = os.path.join(data_path, "state")
    os.makedirs(state_dir, exist_ok=True)
    for model_id, (attendees, subjects, matches) in results.items():
        model_dir = os.path.join(state_dir, model_id.replace("/", "__"))
        os.makedirs(model_dir, exist_ok=True)
        state = {
            "attendees": [_serialize_agent(a) for a in attendees],
            "subjects": [_serialize_agent(s) for s in subjects],
            "matches": {
                agent_id: [m.model_dump() for m in match_list]
                for agent_id, match_list in matches.items()
            },
        }
        with open(os.path.join(model_dir, "simulation.json"), "w") as f:
            json.dump(state, f)


def load_simulation_state(
    model_ids: List[str],
    data_path: str,
    make_model_fn: Callable[[str], object],
    make_scoring_model_fn: Callable[[str], object] = None,
) -> Dict[str, tuple]:
    state_dir = os.path.join(data_path, "state")
    results = {}
    for model_id in model_ids:
        path = os.path.join(state_dir, model_id.replace("/", "__"), "simulation.json")
        with open(path) as f:
            state = json.load(f)
        model = make_model_fn(model_id)
        scoring_model = make_scoring_model_fn(model_id) if make_scoring_model_fn else None
        attendees = [_deserialize_agent(d, model, scoring_model) for d in state["attendees"]]
        subjects = [_deserialize_agent(d, model, scoring_model) for d in state["subjects"]]
        matches = {
            agent_id: [Match(**m) for m in match_list]
            for agent_id, match_list in state["matches"].items()
        }
        results[model_id] = (attendees, subjects, matches)
    return results


def save_lie_state(
    lie_dfs: Dict[str, pd.DataFrame],
    data_path: str,
    ext_lie_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    state_dir = os.path.join(data_path, "state")
    for model_id, df in lie_dfs.items():
        model_dir = os.path.join(state_dir, model_id.replace("/", "__"))
        os.makedirs(model_dir, exist_ok=True)
        df.to_json(os.path.join(model_dir, "lie_df.json"))
    if ext_lie_dfs:
        for model_id, df in ext_lie_dfs.items():
            model_dir = os.path.join(state_dir, model_id.replace("/", "__"))
            os.makedirs(model_dir, exist_ok=True)
            df.to_json(os.path.join(model_dir, "lie_df_ext.json"))


def load_lie_state(
    model_ids: List[str],
    data_path: str,
) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]:
    state_dir = os.path.join(data_path, "state")
    lie_dfs: Dict[str, pd.DataFrame] = {}
    ext_lie_dfs: Dict[str, pd.DataFrame] = {}
    for model_id in model_ids:
        model_dir = os.path.join(state_dir, model_id.replace("/", "__"))
        lie_path = os.path.join(model_dir, "lie_df.json")
        if os.path.exists(lie_path):
            lie_dfs[model_id] = pd.read_json(lie_path)
        ext_path = os.path.join(model_dir, "lie_df_ext.json")
        if os.path.exists(ext_path):
            ext_lie_dfs[model_id] = pd.read_json(ext_path)
    return lie_dfs, ext_lie_dfs if ext_lie_dfs else None


