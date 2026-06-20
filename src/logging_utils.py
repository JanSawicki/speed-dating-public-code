import os
from typing import Dict, List

from langchain_core.messages import SystemMessage

from agent import Agent, Match


def save_conversation_logs(attendees: List[Agent], conversations_path: str):
    for attendee in attendees:
        for conv_key, messages in attendee.conversations.items():
            log = ""
            for message in messages:
                if not isinstance(message, SystemMessage):
                    log += "-------------------------------------------\n"
                    log += f"{message.content}\n"

            filename = f"{attendee.log_name}__{conv_key}.txt"
            with open(os.path.join(conversations_path, filename), "w", encoding="utf-8") as f:
                f.write(log)


def save_match_logs(agents: List[Agent], matches: Dict[str, List[Match]], matches_path: str):
    for agent in agents:
        matches_str = "\n".join(
            f"{m.agent_id} (score: {m.compatibility_score}): {m.explanation}"
            for m in matches[agent.id]
        )
        with open(
            os.path.join(matches_path, f"{agent.log_name}_matches.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(matches_str)
