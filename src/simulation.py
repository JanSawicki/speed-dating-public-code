from typing import Dict, List

from agent import Agent


def build_agents(
    metadata: dict,
    system_prompts: dict,
    model,
    get_matches_prompt: str,
    compatibility_definition: str = "",
    max_messages_per_agent: int = 0,
    scoring_model=None,
) -> List[Agent]:
    return [
        Agent(
            id=agent_id,
            name=info.get("name", ""),
            hobbies=info["hobbies"],
            system_prompt=system_prompts[info["system_prompt"]].format(
                name=info.get("name", ""),
                hobbies="\n".join(f"- {h}" for h in info["hobbies"]),
                compatibility_definition=compatibility_definition,
                max_messages_per_agent=max_messages_per_agent,
            ),
            model=model,
            scoring_model=scoring_model,
            get_matches_prompt=get_matches_prompt,
        )
        for agent_id, info in metadata.items()
    ]


def create_agent_pairs(attendees: List[Agent], subjects: List[Agent]) -> List[tuple]:
    return [
        (attendee, subject)
        for attendee in attendees
        for subject in subjects
        if subject.id.startswith(f"{attendee.id}_SUBJECT_")
    ]


def run_conversations(agent_pairs: List[tuple], initialize_prompt_template: str, max_messages_per_agent: int, conversations_per_pair: int = 1, bar=None):
    for run_id in range(conversations_per_pair):
        for agent_starts, agent_responds in agent_pairs:
            if bar is not None:
                bar.set_postfix(pair=f"{agent_starts.id} → {agent_responds.id}")
            content = agent_starts.initial_message(
                other=agent_responds,
                system_message=initialize_prompt_template,
                run_id=run_id,
            )
            sender, receiver = agent_responds, agent_starts
            for _ in range(max_messages_per_agent * 2):
                content = sender.respond(receiver, content, run_id=run_id)
                sender, receiver = receiver, sender
            if bar is not None:
                bar.update(1)


def collect_matches(agents: List[Agent], conversations_per_pair: int, bar=None) -> Dict[str, list]:
    matches = {}
    for agent in agents:
        all_matches = []
        for run_id in range(conversations_per_pair):
            if bar is not None:
                bar.set_postfix(agent=agent.id, run=run_id)
            run_matches = agent.get_matches(run_id, bar=bar)
            all_matches.extend(run_matches)
        matches[agent.id] = all_matches
    return matches
