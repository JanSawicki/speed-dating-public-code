import argparse
import os
import shutil
import sys

import yaml
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from tqdm.auto import tqdm

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cached_model import CachedModel
from logging_utils import save_conversation_logs, save_match_logs
from state import (
    load_lie_state,
    load_simulation_state,
    save_lie_state,
    save_simulation_state,
)
from analysis import (
    build_ranking_df, save_ranking_heatmap_all_models, save_results_md_all_models,
    build_deception_lift_df, save_deception_lift_chart_all_models, save_deception_lift_md_all_models,
    save_cross_model_heatmap, save_cross_model_md,
    build_variance_df, save_variance_heatmap_all_models, save_variance_md_all_models,
    build_lie_count_df, save_lie_count_heatmap_all_models, save_lie_count_md_all_models,
    save_calibration_curve_all_models, save_calibration_md_all_models,
    build_score_ranking_df, save_score_heatmap_all_models, save_score_md_all_models,
    save_judge_agreement_heatmap_all_models, save_judge_agreement_md_all_models,
    save_pressure_response_chart_all_models, save_pressure_response_md_all_models,
    build_score_distribution_df, save_score_distribution_plot_all_models, save_mass_at_50_md_all_models,
)
from simulation import build_agents, collect_matches, create_agent_pairs, run_conversations


def _make_model(model_id: str, cfg: dict, data_path: str, verbose: bool, max_tokens: int) -> CachedModel:
    sim_cfg = cfg["simulation"]
    cache_dir = os.path.join(data_path, "cache", model_id.replace("/", "__"))
    return CachedModel(
        ChatOpenRouter(model=model_id, max_tokens=max_tokens, temperature=sim_cfg["temperature"]),
        cache_dir=cache_dir,
        cache_batch_size=sim_cfg["cache_batch_size"],
        verbose=verbose,
    )


def run_for_model(model_id: str, cfg: dict, data_path: str, bar=None, verbose: bool = False):
    sim_cfg = cfg["simulation"]
    prompt_cfg = cfg["prompts"]
    agent_cfg = cfg["agents"]

    model_slug = model_id.replace("/", "__")
    conversations_path = os.path.join(data_path, "conversations", model_slug)
    matches_path = os.path.join(data_path, "matches", model_slug)
    for path in [conversations_path, matches_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    cache_dir = os.path.join(data_path, "cache", model_slug)

    max_messages_per_agent = sim_cfg["max_messages_per_agent"]
    conversations_per_pair = sim_cfg["conversations_per_pair"]
    max_tokens_cfg = sim_cfg["max_tokens"]
    temperature = sim_cfg["temperature"]
    cache_batch_size = sim_cfg["cache_batch_size"]
    system_prompts = prompt_cfg["system_prompts"]
    initialize_prompt = prompt_cfg["initialize_conversation_prompt"]
    compatibility_definition = prompt_cfg["compatibility_definition"]
    get_matches_prompt = prompt_cfg["get_matches_prompt"].format(
        compatibility_definition=compatibility_definition
    )

    model = CachedModel(ChatOpenRouter(model=model_id, max_tokens=max_tokens_cfg["conversation"], temperature=temperature), cache_dir=cache_dir, cache_batch_size=cache_batch_size, verbose=verbose)
    scoring_model = CachedModel(ChatOpenRouter(model=model_id, max_tokens=max_tokens_cfg["get_matches"], temperature=temperature), cache_dir=cache_dir, cache_batch_size=cache_batch_size, verbose=verbose)

    attendees = build_agents(agent_cfg["attendees"], system_prompts, model, get_matches_prompt, compatibility_definition, max_messages_per_agent=max_messages_per_agent, scoring_model=scoring_model)
    subjects = build_agents(agent_cfg["subjects"], system_prompts, model, get_matches_prompt, compatibility_definition, max_messages_per_agent=max_messages_per_agent, scoring_model=scoring_model)

    pairs = create_agent_pairs(attendees, subjects)
    if bar is not None:
        bar.set_postfix(model=model_id)
    try:
        run_conversations(pairs, initialize_prompt, max_messages_per_agent, conversations_per_pair, bar=bar)
    finally:
        model.save_caches()

    try:
        matches = collect_matches(attendees, conversations_per_pair, bar=bar)
    finally:
        scoring_model.save_caches()

    save_conversation_logs(attendees, conversations_path)
    save_match_logs(attendees, matches, matches_path)

    return attendees, subjects, matches


def phase_simulation(models: list, cfg: dict, data_path: str, verbose: bool = False) -> dict:
    sim_cfg = cfg["simulation"]
    agent_cfg = cfg["agents"]
    n_pairs = len(agent_cfg["subjects"])
    total = len(models) * sim_cfg["conversations_per_pair"] * n_pairs * 2

    results = {}
    with tqdm(total=total, desc="Conversations") as bar:
        for model_id in models:
            attendees, subjects, matches = run_for_model(model_id, cfg, data_path, bar=bar, verbose=verbose)
            results[model_id] = (attendees, subjects, matches)

    save_simulation_state(results, data_path)
    return results


def phase_lie_detection(
    models: list,
    results: dict,
    judge_model=None,
    lie_detection_model_id: str = None,
) -> tuple:
    total_lie_convs = sum(
        sum(len(a.conversations) for a in attendees if a.hobbies)
        for attendees, _, _ in results.values()
    )

    lie_dfs = {}
    with tqdm(total=total_lie_convs, desc="Lie count") as lie_bar:
        for model_id, (attendees, subjects, _) in results.items():
            lie_bar.set_postfix(model=model_id)
            lie_df = build_lie_count_df(attendees, subjects, bar=lie_bar)
            lie_dfs[model_id] = lie_df

    ext_lie_dfs = None
    if judge_model is not None:
        ext_lie_dfs = {}
        with tqdm(total=total_lie_convs, desc="Lie count (ext judge)") as ext_bar:
            for model_id, (attendees, subjects, _) in results.items():
                ext_bar.set_postfix(model=model_id)
                ext_lie_df = build_lie_count_df(attendees, subjects, judge_model=judge_model, bar=ext_bar)
                ext_lie_dfs[model_id] = ext_lie_df

    return lie_dfs, ext_lie_dfs


def phase_analysis(
    models: list,
    results: dict,
    lie_dfs: dict,
    charts_path: str,
    lie_detection_model_id: str = None,
    ext_lie_dfs: dict = None,
) -> None:
    if os.path.exists(charts_path):
        shutil.rmtree(charts_path)
    os.makedirs(charts_path)

    lift_dfs = {}
    ranking_dfs = {}
    score_dfs = {}
    distribution_dfs = {}
    variance_dfs = {}
    for model_id, (attendees, subjects, matches) in results.items():
        ranking_df = build_ranking_df(attendees, subjects, matches)
        ranking_dfs[model_id] = ranking_df

        score_df = build_score_ranking_df(attendees, subjects, matches)
        score_dfs[model_id] = score_df

        distribution_dfs[model_id] = build_score_distribution_df(attendees, subjects, matches)

        lift_df = build_deception_lift_df(ranking_df)
        lift_dfs[model_id] = lift_df

        variance_df = build_variance_df(attendees, subjects, matches)
        variance_dfs[model_id] = variance_df

    save_calibration_curve_all_models(ranking_dfs, charts_path)
    save_calibration_md_all_models(ranking_dfs, charts_path)
    save_ranking_heatmap_all_models(ranking_dfs, charts_path)
    save_results_md_all_models(ranking_dfs, charts_path)
    save_score_heatmap_all_models(score_dfs, charts_path)
    save_score_md_all_models(score_dfs, charts_path)
    save_score_distribution_plot_all_models(distribution_dfs, charts_path)
    save_mass_at_50_md_all_models(distribution_dfs, charts_path)
    save_variance_heatmap_all_models(variance_dfs, charts_path)
    save_variance_md_all_models(variance_dfs, ranking_dfs, charts_path)
    save_lie_count_md_all_models(lie_dfs, charts_path, ext_lie_dfs=ext_lie_dfs, judge_model_id=lie_detection_model_id)
    save_lie_count_heatmap_all_models(lie_dfs, charts_path, ext_lie_dfs=ext_lie_dfs, judge_model_id=lie_detection_model_id)
    save_deception_lift_md_all_models(lift_dfs, charts_path)
    save_deception_lift_chart_all_models(lift_dfs, charts_path)
    save_cross_model_heatmap(ranking_dfs, charts_path)
    save_cross_model_md(ranking_dfs, charts_path)
    save_pressure_response_chart_all_models(ranking_dfs, charts_path)
    save_pressure_response_md_all_models(ranking_dfs, charts_path)

    if lie_dfs and ext_lie_dfs:
        save_judge_agreement_heatmap_all_models(
            lie_dfs, ext_lie_dfs, charts_path,
            judge_model_id=lie_detection_model_id,
        )
        save_judge_agreement_md_all_models(
            lie_dfs, ext_lie_dfs, charts_path,
            judge_model_id=lie_detection_model_id,
        )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Speed dating simulation")
    parser.add_argument(
        "--config",
        default=os.path.join(_root_dir, "config", "config.yaml"),
        help="Path to config YAML file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--phase",
        choices=["simulation", "lie-detection", "analysis", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print cache load/save information",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = os.getenv("DATA_PATH")
    models = cfg["models"]
    sim_cfg = cfg["simulation"]
    lie_detection_model_id = cfg.get("lie_detection_model")
    charts_path = os.path.join(data_path, "charts")

    judge_model = None
    if lie_detection_model_id:
        judge_cache_dir = os.path.join(data_path, "cache", lie_detection_model_id.replace("/", "__"))
        judge_model = CachedModel(
            ChatOpenRouter(model=lie_detection_model_id, max_tokens=sim_cfg["max_tokens"]["lie_detection"], temperature=sim_cfg["temperature"]),
            cache_dir=judge_cache_dir,
            cache_batch_size=sim_cfg["cache_batch_size"],
            verbose=args.verbose,
        )

    if args.phase in ("simulation", "all"):
        results = phase_simulation(models, cfg, data_path, verbose=args.verbose)
    else:
        results = load_simulation_state(
            models, data_path,
            lambda mid: _make_model(mid, cfg, data_path, args.verbose, sim_cfg["max_tokens"]["conversation"]),
            lambda mid: _make_model(mid, cfg, data_path, args.verbose, sim_cfg["max_tokens"]["get_matches"]),
        )

    if args.phase in ("lie-detection", "all"):
        lie_dfs, ext_lie_dfs = phase_lie_detection(
            models, results,
            judge_model=judge_model,
            lie_detection_model_id=lie_detection_model_id,
        )
        save_lie_state(lie_dfs, data_path, ext_lie_dfs=ext_lie_dfs)
    else:
        lie_dfs, ext_lie_dfs = load_lie_state(models, data_path)

    if args.phase in ("analysis", "all"):
        phase_analysis(
            models, results, lie_dfs, charts_path,
            lie_detection_model_id=lie_detection_model_id,
            ext_lie_dfs=ext_lie_dfs,
        )


if __name__ == "__main__":
    main()
