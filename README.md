# speed-dating-public-code

Experimental pipeline for a simulated speed-dating study of LLM deception under social
pressure. Agents converse, an attendee agent rates each subject's compatibility, and a
suite of analysis modules turns the results into 3×4 (compatibility tier × pressure
level) tables and charts.

This is the code-only companion to a research paper; it does not include the paper
text, generated charts, or any cached model output / experiment data from the original
study.

## Project structure

```
config/
  config.yaml          # Full config: models, simulation settings, prompts, agents
  config_small.yaml     # Reduced subset for fast test runs
src/
  main.py               # CLI entry point; orchestrates all phases
  agent.py              # Agent class — conversation history, LLM calls, get_matches()
  simulation.py         # build_agents(), create_agent_pairs(), run_conversations(), collect_matches()
  cached_model.py       # CachedModel — wraps ChatOpenRouter with a disk-backed response cache
  cache.py              # DiskCache implementation
  state.py              # Serialise/deserialise agents + matches to JSON
  logging_utils.py      # save_conversation_logs(), save_match_logs()
  analysis/             # Per-metric analysis + chart-building modules
```

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Create a `.env` file in the project root:

```
DATA_PATH=./data
OPENROUTER_API_KEY=<your key from openrouter.ai>
```

## Running

The pipeline runs in three independently re-runnable phases:

```bash
uv run python src/main.py --phase simulation     # run conversations, collect compatibility scores
uv run python src/main.py --phase lie-detection  # self-eval + external-judge lie counting
uv run python src/main.py --phase analysis       # build charts and tables from saved state
uv run python src/main.py --phase all            # run all three phases
```

Use the smaller config for a fast sanity check:

```bash
uv run python src/main.py --config config/config_small.yaml
```

## Configuration

All settings live in `config/config.yaml`:

- `models` — list of OpenRouter model IDs used as conversational agents
- `lie_detection_model` — OpenRouter model ID for the external lie-detection judge
- `simulation` — turn limits, runs per pair, temperature, token limits
- `prompts` — system prompts per pressure level, the compatibility definition, and the
  match-assessment prompt
- `agents` — attendee and subject agent definitions

Subject agent IDs must follow the `SUBJECT_L{level}_{tier}` convention (tier one of
`COMPATIBLE`, `PARTIAL`, `INCOMPATIBLE`) for the analysis modules to parse pressure
level and tier correctly from the agent ID.

## Output layout

Output is written under `$DATA_PATH/`:

| Path | Contents |
|---|---|
| `conversations/<model>/` | Per-pair conversation logs |
| `matches/<model>/` | Match decisions with explanations |
| `cache/<model>/` | Disk cache for LLM responses |
| `state/<model>/simulation.json` | Serialised agents + matches |
| `state/<model>/lie_df.json` | Self-evaluation lie counts |
| `state/<model>/lie_df_ext.json` | External judge lie counts |
| `charts/` | All generated charts (PDF + Markdown tables) |

## Caching

All API calls are cached to disk via `CachedModel` (`src/cached_model.py`). Re-running
the pipeline against an existing cache reproduces identical outputs with no additional
API cost. Delete the relevant subdirectory under `data/cache/` to force fresh calls.

## License

MIT.
