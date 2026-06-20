import re
from typing import Dict, List

from agent import Agent, Match

TIERS = ["COMPATIBLE", "PARTIAL", "INCOMPATIBLE"]

_SUBJECT_RE = re.compile(r"^ATTENDEE_(\d+)_SUBJECT_L(\d+)_(.+)$")


def _parse_subject_id(subject_id: str):
    m = _SUBJECT_RE.match(subject_id)
    return int(m.group(2)), m.group(3)


def _parse_attendee_from_subject_id(subject_id: str) -> str:
    m = _SUBJECT_RE.match(subject_id)
    return f"ATTENDEE_{int(m.group(1)):02d}"


def _base_subject_id(agent_id: str) -> str:
    return agent_id.split("__run")[0]


def _subject_columns(subjects: List[Agent]) -> List[str]:
    levels = sorted({_parse_subject_id(s.id)[0] for s in subjects})
    return [f"Level {l}" for l in levels]


def _iter_matches(attendees: List[Agent], subjects: List[Agent], matches: Dict[str, List[Match]]):
    attendee_ids = {a.id for a in attendees}
    subject_meta = {s.id: _parse_subject_id(s.id) for s in subjects}
    for agent_id, match_list in matches.items():
        if agent_id not in attendee_ids:
            continue
        for match in match_list:
            subject_id = _base_subject_id(match.agent_id)
            if subject_id not in subject_meta:
                continue
            level, tier = subject_meta[subject_id]
            yield agent_id, tier, f"Level {level}", match


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")
