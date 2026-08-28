"""Factory functions for sample generation."""
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from synth.ts_generator.generate import (
    all_attribute_set,
    generate_controlled_attributes,
    generate_random_attributes,
    generate_time_series,
)
from synth.ts_generator.utils.common_utils import (
    article,
    DEFAULT_LOCAL_FEATURE_CONFIG,
    DEFAULT_SHAPE_FEATURE_CONFIG,
    BatchSampleGenerator,
    get_random_change_position,
    select_metrics_from_clusters,
    generate_threshold_for_sample,
    DEFAULT_THRESHOLD,
)
from synth.ts_generator.utils.probability_utils import select_weighted_qa_index
from synth.align.utils import metric_to_controlled_attributes, timeseries_encoding
from synth.align.config import (
    Config,
    Difficulty,
    Mode,
    PromptIndexer,
    SampleResult,
)
from synth.align.generators import (
    UTSQAGenerator,
    MTSLocalQAGenerator,
    MTSShapeQAGenerator,
)


def _resolve_threshold(seq_len: int) -> Tuple[int, bool]:
    """Resolve threshold for a sample: generate or fall back to default.

    Returns (threshold, threshold_provided).
    """
    threshold = generate_threshold_for_sample(seq_len)
    threshold_provided = threshold is not None
    if not threshold_provided:
        threshold = DEFAULT_THRESHOLD
    return threshold, threshold_provided


def select_metrics_for_generation(
    metric_config: List[Dict],
    mode: Mode,
    target_positive_size: Optional[int] = None,
) -> Tuple[str, List[List[str]], List[str], Dict[str, str]]:
    # When targeting a specific cluster size, retry until we find a category
    # with enough total metrics.
    for _ in range(10):
        sample = random.choice(metric_config)
        total_metrics = sum(len(ms) for ms in sample['cluster'].values())
        if target_positive_size is None or total_metrics >= target_positive_size:
            break

    situation = sample['category']
    cluster = sample['cluster']
    metric_to_cluster = {m: c for c, ms in cluster.items() for m in ms}

    if target_positive_size is not None:
        positive_clusters, negative_metrics, _ = select_metrics_from_clusters(
            cluster,
            target_positive_size=target_positive_size,
            max_negative=5,
        )
    else:
        num_positive_clusters = random.randint(1, 3)
        positive_clusters, negative_metrics, _ = select_metrics_from_clusters(
            cluster,
            num_positive_clusters=num_positive_clusters,
            max_negative=5,
        )

    return situation, positive_clusters, negative_metrics, metric_to_cluster


def select_single_qa(
    result: SampleResult,
    qa_type_weights: Dict[str, float],
    eval_type_weights: Optional[Dict[str, float]] = None,
) -> None:
    if not result.questions:
        return

    eval_types = result.eval_tasks if result.eval_tasks else None
    idx = select_weighted_qa_index(
        result.qa_types, qa_type_weights,
        eval_types=eval_types,
        eval_type_weights=eval_type_weights,
    )

    # How many prompts precede the selected Q&A pair
    prompt_offset = sum(len(result.llm_prompts[i]) for i in range(idx))
    num_prompts = len(result.llm_prompts[idx])

    # Re-index placeholders in the selected answer: old indices -> 0-based
    answer = result.answers[idx]

    # Two passes to avoid collisions (e.g. prompt3 -> prompt0 before prompt0 is renamed)
    for new_idx in range(num_prompts):
        old_idx = prompt_offset + new_idx
        answer = answer.replace(f"<|prompt{old_idx}|>", f"<|__tmp{new_idx}__|>")
    for new_idx in range(num_prompts):
        answer = answer.replace(f"<|__tmp{new_idx}__|>", f"<|prompt{new_idx}|>")

    # Keep only the selected Q&A pair
    result.questions = [result.questions[idx]]
    result.answers = [answer]
    result.llm_prompts = [result.llm_prompts[idx]]
    result.fields = [result.fields[idx]]
    result.qa_types = [result.qa_types[idx]]

    # Propagate eval signals for the selected pair
    if result.eval_tasks:
        result.eval_task = result.eval_tasks[idx]
        result.eval_metadata = result.eval_metadatas[idx] if idx < len(result.eval_metadatas) else {}
        result.eval_tasks = [result.eval_tasks[idx]]
        result.eval_metadatas = [result.eval_metadatas[idx]] if idx < len(result.eval_metadatas) else [{}]


def generate_uts_sample(
    config: Config,
    indexer: PromptIndexer,
    seq_len: int,
    difficulty: Difficulty = Difficulty.EASY,
) -> SampleResult:
    threshold, threshold_provided = _resolve_threshold(seq_len)

    # Select metric
    sample = random.choice(config.metric_config)
    category, metric = sample["category"], random.choice(sample["metrics"])

    # Generate attributes & time series
    metric_attrs = metric_to_controlled_attributes(metric)
    if config.disable_metric_config:
        attribute_pool = generate_random_attributes(
            all_attribute_set["overall_attribute"],
            all_attribute_set["change"],
            seq_len=seq_len
        )
    else:
        attribute_pool = generate_controlled_attributes(
            metric_attrs,
            seq_len=seq_len
        )
    timeseries, attribute_pool = generate_time_series(attribute_pool, seq_len)

    attribute_pool["metric_name"] = metric
    attribute_pool["situation"] = category
    scaled_ts, ts_prompt, _ = timeseries_encoding(timeseries, config.encoding_method)

    # Build base prompt
    base_prompt = (
        f"You are a time series analysis expert. This is a metric called {metric} "
        f"collected from {category} with length of {seq_len}: {ts_prompt}."
    )

    # Generate QA
    qa_gen = UTSQAGenerator(
        config=config,
        indexer=indexer,
        timeseries=timeseries,
        attributes=attribute_pool,
        metric=metric,
        category=category,
        seq_len=seq_len,
        threshold=threshold,
        threshold_provided=threshold_provided,
        difficulty=difficulty
    )
    r = qa_gen.generate_all_qa()
    questions, answers, llm_prompts, fields, qa_types = r.questions, r.answers, r.llm_prompts, r.fields, r.qa_types
    eval_tasks, eval_metadatas = r.eval_tasks, r.eval_metadatas

    # Build label
    label = {
        'mode': Mode.UTS.value,
        'metrics': [metric],
        'situation': category,
        'attribute_pool': [attribute_pool],
        'clusters': [],
        'correlations': [],
        'position': None,
    }

    return SampleResult(
        mode=Mode.UTS,
        original_timeseries=[timeseries],
        encoded_timeseries=[scaled_ts],
        metrics=[metric],
        attributes=[attribute_pool],
        base_prompt=base_prompt,
        questions=questions,
        answers=answers,
        llm_prompts=llm_prompts,
        fields=fields,
        corr_pool=[],
        label=label,
        qa_types=qa_types,
        eval_tasks=eval_tasks,
        eval_metadatas=eval_metadatas,
        situation=category
    )


def _shuffle_and_encode_mts(
    all_ts, all_attrs, all_metrics, all_idx,
    situation, seq_len, encoding_method,
    extra_lists=None,
):
    """Shuffle all MTS lists in lockstep and encode time series into a base prompt.

    Returns (combined_ts, combined_attrs, combined_metrics, combined_idx,
             original_ts, encoded_ts, base_prompt, extra_combined)
    where extra_combined mirrors extra_lists after the same shuffle.
    """
    total = len(all_ts)
    shuffle_order = np.random.permutation(total)
    combined_ts = [all_ts[i] for i in shuffle_order]
    combined_attrs = [all_attrs[i] for i in shuffle_order]
    combined_metrics = [all_metrics[i] for i in shuffle_order]
    combined_idx = [all_idx[i] for i in shuffle_order]
    extra_combined = [[lst[i] for i in shuffle_order] for lst in (extra_lists or [])]

    original_ts = copy.deepcopy(combined_ts)

    encoded_ts = []
    base_prompt = f"In {article(situation)} {situation} system, there are {len(combined_ts)} metrics:"
    for ts, metric in zip(combined_ts, combined_metrics):
        scaled_ts, ts_prompt, _ = timeseries_encoding(ts, encoding_method)
        encoded_ts.append(scaled_ts)
        base_prompt += f"\n {metric} is of length {seq_len}: {ts_prompt};"

    return combined_ts, combined_attrs, combined_metrics, combined_idx, original_ts, encoded_ts, base_prompt, extra_combined


def generate_mts_local_sample(
    config: Config,
    indexer: PromptIndexer,
    seq_len: int,
    difficulty: Difficulty = Difficulty.EASY
) -> SampleResult:
    threshold, threshold_provided = _resolve_threshold(seq_len)

    # Select metrics
    situation, positive_clusters, negative_metrics, metric_to_cluster = select_metrics_for_generation(
        config.metric_config, Mode.MTS_LOCAL
    )
    positive_metrics = [m for cluster in positive_clusters for m in cluster]

    # Generate time series
    ts_generator = BatchSampleGenerator(
        mode="local",
        feature_config=DEFAULT_LOCAL_FEATURE_CONFIG,
        max_retries=10000
    )

    # Generate positive series for each cluster
    positive_timeseries = []
    positive_attributes = []
    positive_idx_list = []
    positive_change_positions = []

    for i, cluster_metrics in enumerate(positive_clusters):
        # Find a change position spaced apart from previous ones.
        # Relax the spacing constraint after MAX_RETRIES to avoid
        # infinite loops with short seq_len (e.g. 16).
        MAX_RETRIES = 200
        min_spacing = seq_len // 5
        for attempt in range(MAX_RETRIES):
            change_position = get_random_change_position(seq_len)
            if all(abs(change_position - pos) > min_spacing for pos in positive_change_positions):
                break
        else:
            # Relax: accept any position that doesn't exactly collide
            for _ in range(MAX_RETRIES):
                change_position = get_random_change_position(seq_len)
                if all(abs(change_position - pos) > 1 for pos in positive_change_positions):
                    break

        cur_ts, cur_attr, actual_pos, _ = ts_generator.generate_positive_timeseries(
            len(cluster_metrics), seq_len, change_position,
            validate_local_count=True, threshold=threshold
        )
        positive_timeseries.extend(cur_ts)
        positive_attributes.extend(cur_attr)
        positive_idx_list.extend([i] * len(cluster_metrics))
        positive_change_positions.append(actual_pos)

    # Generate negative series
    neg_ts, neg_attr, _, _ = ts_generator.generate_negative_timeseries(
        len(negative_metrics),
        positive_change_positions,
        seq_len,
        threshold=threshold,
        validate_local_count=True
    )

    # Combine, shuffle, and encode
    all_ts = positive_timeseries + neg_ts
    all_attrs = positive_attributes + neg_attr
    all_metrics = positive_metrics + negative_metrics
    all_idx = positive_idx_list + [None] * len(negative_metrics)

    (combined_ts, combined_attrs, combined_metrics, combined_idx,
     original_ts, encoded_ts, base_prompt, _) = _shuffle_and_encode_mts(
        all_ts, all_attrs, all_metrics, all_idx,
        situation, seq_len, config.encoding_method,
    )

    # Generate QA
    qa_gen = MTSLocalQAGenerator(
        config=config,
        indexer=indexer,
        situation=situation,
        metrics=combined_metrics,
        attributes=combined_attrs,
        cluster_indices=combined_idx,
        positive_clusters=positive_clusters,
        change_positions=positive_change_positions,
        metric_to_cluster=metric_to_cluster,
        original_timeseries=original_ts,
        threshold=threshold,
        threshold_provided=threshold_provided,
        seq_len=seq_len,
        difficulty=difficulty
    )
    r = qa_gen.generate_all_qa()
    questions, answers, llm_prompts, fields, qa_types = r.questions, r.answers, r.llm_prompts, r.fields, r.qa_types
    correlations, clusters, corr_pool = r.correlations, r.clusters, r.corr_pool_list
    eval_tasks, eval_metadatas = r.eval_tasks, r.eval_metadatas
    
    # Build label
    label = {
        'mode': Mode.MTS_LOCAL.value,
        'metrics': combined_metrics,
        'situation': situation,
        'attribute_pool': combined_attrs,
        'clusters': clusters,
        'correlations': correlations,
        'position': int(positive_change_positions[0]) if positive_change_positions else 0,
    }

    return SampleResult(
        mode=Mode.MTS_LOCAL,
        original_timeseries=original_ts,
        encoded_timeseries=encoded_ts,
        metrics=combined_metrics,
        attributes=combined_attrs,
        base_prompt=base_prompt,
        questions=questions,
        answers=answers,
        llm_prompts=llm_prompts,
        fields=fields,
        corr_pool=corr_pool,
        label=label,
        qa_types=qa_types,
        eval_tasks=eval_tasks,
        eval_metadatas=eval_metadatas,
        situation=situation
    )


def generate_mts_shape_sample(
    config: Config,
    indexer: PromptIndexer,
    seq_len: int,
    difficulty: Difficulty = Difficulty.EASY,
    target_cluster_size: Optional[int] = None,
    target_anticluster_size: Optional[int] = None,
) -> SampleResult:
    threshold, threshold_provided = _resolve_threshold(seq_len)

    # Select metrics
    situation, positive_clusters, negative_metrics, metric_to_cluster = select_metrics_for_generation(
        config.metric_config, Mode.MTS_SHAPE,
        target_positive_size=target_cluster_size,
    )
    positive_metrics = [m for cluster in positive_clusters for m in cluster]

    # Generate time series
    ts_generator = BatchSampleGenerator(
        mode="shape",
        feature_config=DEFAULT_SHAPE_FEATURE_CONFIG
    )

    # Generate positive series for each cluster
    positive_timeseries = []
    positive_attributes = []
    positive_idx_list = []
    cluster_points = []
    all_points = []

    for i, cluster_metrics in enumerate(positive_clusters):
        t, a, base_pts, pts_list = ts_generator.generate_positive_timeseries(
            len(cluster_metrics), seq_len, threshold=threshold
        )
        positive_timeseries.extend(t)
        positive_attributes.extend(a)
        positive_idx_list.extend([i] * len(cluster_metrics))
        cluster_points.append(base_pts)
        all_points.extend(pts_list)

    # --- Anti-trend cluster (repurpose some negatives) ---
    anti_trend_ts, anti_trend_attrs, anti_trend_points = [], [], []
    anti_trend_metrics = []
    remaining_negatives = list(negative_metrics)

    if negative_metrics and target_anticluster_size != 0:
        # Try each positive cluster until we find one with non-steady trends
        for base_pts in cluster_points:
            if target_anticluster_size is not None:
                anti_count = min(len(negative_metrics), target_anticluster_size)
            else:
                # Aim for ~50% of negatives to be anti-trend (supports anti_judgment balance)
                half_neg = max(2, len(negative_metrics) // 2)
                anti_count = min(len(negative_metrics), random.randint(2, half_neg))
            anti_ts, anti_attrs, anti_pts = ts_generator.generate_anti_trend_timeseries(
                base_pts, anti_count, seq_len, threshold=threshold
            )
            if anti_ts:  # Non-empty = base had non-steady trends
                anti_trend_ts = anti_ts
                anti_trend_attrs = anti_attrs
                anti_trend_points = anti_pts
                anti_trend_metrics = negative_metrics[:anti_count]
                remaining_negatives = negative_metrics[anti_count:]
                break

    # Generate remaining negatives normally
    ref_points = cluster_points[0] if cluster_points else []
    if remaining_negatives:
        neg_ts, neg_attr, neg_diff, neg_points = ts_generator.generate_negative_timeseries(
            len(remaining_negatives), ref_points, seq_len, threshold=threshold
        )
    else:
        neg_ts, neg_attr, neg_diff, neg_points = [], [], [], []

    # Combine, shuffle, and encode
    all_ts = positive_timeseries + anti_trend_ts + neg_ts
    all_attrs = positive_attributes + anti_trend_attrs + neg_attr
    all_metrics = positive_metrics + anti_trend_metrics + remaining_negatives
    all_idx = positive_idx_list + [None] * len(anti_trend_metrics) + [None] * len(remaining_negatives)
    all_pts = all_points + anti_trend_points + (neg_points if neg_points else [])

    (combined_ts, combined_attrs, combined_metrics, combined_idx,
     original_ts, encoded_ts, base_prompt, [combined_points]) = _shuffle_and_encode_mts(
        all_ts, all_attrs, all_metrics, all_idx,
        situation, seq_len, config.encoding_method,
        extra_lists=[all_pts],
    )

    # Generate QA
    qa_gen = MTSShapeQAGenerator(
        config=config,
        indexer=indexer,
        situation=situation,
        metrics=combined_metrics,
        attributes=combined_attrs,
        cluster_indices=combined_idx,
        points_list=combined_points,
        metric_to_cluster=metric_to_cluster,
        seq_len=seq_len,
        threshold=threshold,
        threshold_provided=threshold_provided,
        difficulty=difficulty,
        original_timeseries=original_ts,
    )
    r = qa_gen.generate_all_qa()

    # Build label
    label = {
        'mode': Mode.MTS_SHAPE.value,
        'metrics': combined_metrics,
        'situation': situation,
        'attribute_pool': combined_attrs,
        'clusters': r.clusters,
        'correlations': r.correlations,
        'position': None,
    }

    return SampleResult(
        mode=Mode.MTS_SHAPE,
        original_timeseries=original_ts,
        encoded_timeseries=encoded_ts,
        metrics=combined_metrics,
        attributes=combined_attrs,
        base_prompt=base_prompt,
        questions=r.questions,
        answers=r.answers,
        llm_prompts=r.llm_prompts,
        fields=r.fields,
        corr_pool=r.corr_pool_list,
        label=label,
        qa_types=r.qa_types,
        eval_tasks=r.eval_tasks,
        eval_metadatas=r.eval_metadatas,
        situation=situation
    )


def generate_sample(
    mode: Mode,
    config: Config,
    indexer: PromptIndexer,
    seq_len: int,
    difficulty: Difficulty = Difficulty.EASY,
    debug: bool = False,
    target_cluster_size: Optional[int] = None,
    target_anticluster_size: Optional[int] = None,
) -> SampleResult:
    if mode == Mode.UTS:
        result = generate_uts_sample(config, indexer, seq_len, difficulty=difficulty)
    elif mode == Mode.MTS_LOCAL:
        result = generate_mts_local_sample(config, indexer, seq_len, difficulty=difficulty)
    elif mode == Mode.MTS_SHAPE:
        result = generate_mts_shape_sample(config, indexer, seq_len, difficulty=difficulty,
                                           target_cluster_size=target_cluster_size,
                                           target_anticluster_size=target_anticluster_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not debug:
        # Keep only one Q&A pair per data item using two-level weighted sampling
        select_single_qa(result, config.qa_type_weights, config.eval_type_weights)
    return result
