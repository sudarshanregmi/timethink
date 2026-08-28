"""Main entry point for QA dataset generation.

Usage:
    python -m synth.align
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from transformers import AutoTokenizer

from synth.align.config import Config
from synth.align.dataset import (
    DatasetGenerator,
    FILTER_GROUPS,
    write_split_files,
    split_results,
)
from synth.align.evol.generate_tsevol import evol_instruct, evol_instruct_paraphrase
from synth.align.token_filter import filter_by_token_length
from synth.utils.llm_utils import LLMClient


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="QA Dataset Generator")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: exhaustively generate exactly one sample per unique question sub-type.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Skip LLM fill-in and use placeholder answers instead.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip TS context from input field (keep only the question after the last <ts>).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        metavar="TYPES",
        help=(
            "Comma-separated QA type filter (implies --debug --dryrun). "
            "Groups: " + ", ".join(sorted(FILTER_GROUPS.keys())) + ". "
            "Or exact eval types like stat_numerical, segment_judgment, etc."
        ),
    )
    args = parser.parse_args()

    # --filter implies --debug --dryrun
    if args.filter:
        args.debug = True
        args.dryrun = True

    # --filter ood is special: OOD types are generated post-hoc from base data.
    # Strip 'ood' from the filter so base generation runs unfiltered.
    _ood_only_filter = False
    if args.filter:
        ood_group = FILTER_GROUPS.get('ood', frozenset())
        filter_parts = [f.strip() for f in args.filter.split(',')]
        non_ood = [f for f in filter_parts if f != 'ood' and f not in ood_group]
        if not non_ood and any(f == 'ood' or f in ood_group for f in filter_parts):
            _ood_only_filter = True
            args.filter = None  # let base generation run unfiltered

    # Configure loguru: suppress DEBUG logs in normal runs
    logger.remove()
    verbose = args.debug
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")

    random.seed(42)
    np.random.seed(42)

    # Load configuration
    config = Config.from_yaml()

    # CLI flags override YAML values
    if args.debug:
        config.debug = True
    if args.dryrun:
        config.dryrun = True
    if args.strip:
        config.strip = True
    if args.filter:
        config.debug_filter = [f.strip() for f in args.filter.split(',')]
    print("=" * 60)
    print("Unified LLM QA Generator")
    print("=" * 60)
    if config.debug:
        print("*** DEBUG MODE: one sample per unique question sub-type (exhaustive) ***")
    print(f"Total samples: {config.num_data}")
    if not config.dryrun:
        print(f"  Seeds: {config.num_data} total, TSEvol target: {config.num_data_tsevol} replacements ({config.tsevol_ratio:.0%})")
    print(f"Sequence length — main pool: 50% default=256, 50% uniform[16, {config.main_pool_max_seq_len}].")
    if config.main_pool_max_seq_len <= 256:
        print(f"                 Main pool is curriculum-capped at 256. Long-seq data")
        print(f"                 comes from length_extension (train_rl + val + OOD test")
        print(f"                 buckets). See synth/align/length_extension.py.")
    else:
        print(f"                 main_pool_max_seq_len={config.main_pool_max_seq_len} — curriculum disabled.")
    print(f"Encoding method: {config.encoding_method}")
    print(f"Output directory: {config.output_base_dir}")
    print(f"Dry run: {config.dryrun}")
    print(f"Max token length: {config.max_token_length} (tokenizer: {config.tokenizer_path})")
    print("=" * 60)

    # Ensure output directory exists
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load training tokenizer for token length filtering
    print(f"Loading tokenizer from: {config.tokenizer_path}")
    train_tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path, trust_remote_code=True
    )

    # Create LLM workers once (reused by both seed generation and TSEvol)
    llm_client = None
    try:
        if not config.dryrun:
            llm_client = LLMClient(
                model_path=config.local_llm_path, engine='vllm',
                ctx_length=config.ctx_length,
            )
            llm_client.wait_for_ready()

        # Generate dataset
        generator = DatasetGenerator(config, llm_client=llm_client)
        results = generator.generate()

        print(f"\nTotal samples generated: {len(results)}")

        # Balance binary verdict types (hard 50/50 at the source)
        from synth.align.dataset import balance_binary_results
        results = balance_binary_results(results)
        print(f"After balance: {len(results)} samples")

        # Filter seeds exceeding max token length
        results = filter_by_token_length(
            results, train_tokenizer, config.max_token_length, stage="seed"
        )

        # Default all samples to tsevol=False; TSEvol will overwrite to True
        for r in results:
            if r.eval_metadata is None:
                r.eval_metadata = {}
            r.eval_metadata["tsevol"] = False

        # Generate OOD evaluation questions (eval-only, never in training)
        from synth.align.generators.ood_eval import generate_ood_for_split

        def _apply_full_evol(split_data, evolved):
            """Replace seed QA with full-evolved QA and retag eval_type='tsevol'."""
            for seed_idx, evol_data in evolved.items():
                r = split_data[seed_idx]
                r.questions[0] = evol_data["question"]
                r.answers[0] = evol_data["answer"]
                r.eval_task = "tsevol"
                r.eval_tasks = ["tsevol"]
                if r.eval_metadata is None:
                    r.eval_metadata = {}
                r.eval_metadata["evol_strategy"] = evol_data["strategy"]
                r.eval_metadata["tsevol"] = True
                if len(r.eval_metadatas) > 0:
                    r.eval_metadatas[0] = r.eval_metadata
                else:
                    r.eval_metadatas = [r.eval_metadata]

        def _apply_paraphrase(split_data, evolved):
            """Replace seed QUESTION only; keep seed answer + eval_type intact.

            Paraphrase samples stay routed to the original rule-based reward
            scorer (no LLM-generated labels), so the reward signal remains
            verifiable. Metadata records that this sample was paraphrased.
            """
            for seed_idx, evol_data in evolved.items():
                r = split_data[seed_idx]
                r.questions[0] = evol_data["question"]
                # r.answers[0] intentionally unchanged
                if r.eval_metadata is None:
                    r.eval_metadata = {}
                r.eval_metadata["evol_strategy"] = "paraphrase"
                r.eval_metadata["tsevol"] = True
                if len(r.eval_metadatas) > 0:
                    r.eval_metadatas[0] = r.eval_metadata
                else:
                    r.eval_metadatas = [r.eval_metadata]

        # Write output files
        print("\nWriting output files...")
        max_ood = config.max_ood_samples

        if config.debug:
            # Debug mode: single bucket; TSEvol (full) on all if LLM available.
            if not config.dryrun:
                print("\nRunning TS Evolution (debug, full mode on all)...")
                evolved = evol_instruct(config, results, llm_client=llm_client)
                _apply_full_evol(results, evolved)
                results = filter_by_token_length(
                    results, train_tokenizer, config.max_token_length,
                    stage="post-tsevol",
                )
            ood_samples = generate_ood_for_split(results, max_ood_samples=max_ood)
            print(f"  - OOD eval:       {len(ood_samples)}")
            if _ood_only_filter:
                results = ood_samples
            else:
                results.extend(ood_samples)
            write_split_files(config, "debug", results)
        else:
            # Production: split FIRST, then per-split TSEvol with different modes.
            # This guarantees: (a) RL split gets ONLY paraphrase (verifiable
            # reward, no label-noise hacking surface), (b) the SFT-stage
            # checkpoint doubles as the SFT-only baseline without needing a
            # separate run — it trains on exactly the labeled budget the
            # RL split pretends not to have (Option A).
            print("Splitting data: train/val/test, train → sft/rl...")
            sft_data, rl_data, val_data, test_data, _train_union = split_results(results)

            # Per-split TSEvol. Runs in both modes:
            #   - production (llm_client set): real LLM rewrites
            #   - dryrun (llm_client=None): stub mode — picks seeds + tags
            #     metadata without LLM, so routing / counts / eval_type
            #     preservation / RL-isolation are validated at full scale.
            print("\n" + "=" * 60)
            mode_label = "stub (no LLM)" if llm_client is None else "full (LLM)"
            print(f"Running per-split TS Evolution — {mode_label}")
            print("=" * 60)
            tsevol_ratio = config.tsevol_ratio

            # SFT (100K target): 50/50 full evolution + paraphrase
            sft_budget = int(len(sft_data) * tsevol_ratio)
            sft_full_target = sft_budget // 2
            sft_para_target = sft_budget - sft_full_target
            if sft_full_target > 0:
                print(f"\n[SFT] Full evolution target: {sft_full_target}")
                evolved = evol_instruct(
                    config, sft_data,
                    llm_client=llm_client,
                    target_count=sft_full_target,
                )
                _apply_full_evol(sft_data, evolved)
            if sft_para_target > 0:
                print(f"\n[SFT] Paraphrase target: {sft_para_target}")
                # Only paraphrase seeds not already full-evolved
                unused_idx = [
                    i for i, r in enumerate(sft_data)
                    if not r.eval_metadata.get("tsevol")
                ]
                subset = [sft_data[i] for i in unused_idx]
                evolved_sub = evol_instruct_paraphrase(
                    subset, sft_para_target, llm_client=llm_client,
                )
                # Map subset indices back to sft_data indices
                remapped = {
                    unused_idx[sub_idx]: data
                    for sub_idx, data in evolved_sub.items()
                }
                _apply_paraphrase(sft_data, remapped)

            # RL (50K target): 100% paraphrase, 0% full evolution
            rl_target = int(len(rl_data) * tsevol_ratio)
            if rl_target > 0:
                print(f"\n[RL] Paraphrase target: {rl_target} (full evolution: 0)")
                evolved = evol_instruct_paraphrase(
                    rl_data, rl_target, llm_client=llm_client,
                )
                _apply_paraphrase(rl_data, evolved)

            # Val & Test: paraphrase for phrasing-robustness in reporting
            for name, data in [("Val", val_data), ("Test", test_data)]:
                target = int(len(data) * tsevol_ratio)
                if target > 0:
                    print(f"\n[{name}] Paraphrase target: {target}")
                    evolved = evol_instruct_paraphrase(
                        data, target, llm_client=llm_client,
                    )
                    _apply_paraphrase(data, evolved)

            # Filter each split post-TSEvol
            sft_data = filter_by_token_length(
                sft_data, train_tokenizer, config.max_token_length,
                stage="post-tsevol:sft",
            )
            rl_data = filter_by_token_length(
                rl_data, train_tokenizer, config.max_token_length,
                stage="post-tsevol:rl",
            )
            val_data = filter_by_token_length(
                val_data, train_tokenizer, config.max_token_length,
                stage="post-tsevol:val",
            )
            test_data = filter_by_token_length(
                test_data, train_tokenizer, config.max_token_length,
                stage="post-tsevol:test",
            )

            # Option A (no RL-answer leak): the SFT-only baseline IS the SFT
            # phase of the pipeline. `sft.sh` trains on `train_sft.parquet`;
            # its checkpoint serves both as RL init AND as the baseline for
            # comparison. No separate `sft_only.sh` or `train_sft_only.parquet`.

            ood_samples = generate_ood_for_split(test_data, max_ood_samples=max_ood)
            print(f"\n  - Train SFT:      {len(sft_data)}  (also the SFT-only baseline pool)")
            print(f"  - Train RL:       {len(rl_data)}")
            print(f"  - Val:            {len(val_data)}")
            print(f"  - Test:           {len(test_data)} + {len(ood_samples)} OOD")
            test_data.extend(ood_samples)

            # ----------------------------------------------------------
            # Length-extension phase (Option C curriculum, 2026-04-21).
            #
            # The main pool is ≤256-heavy because chunked generators were
            # gated at 256. We keep that SFT-friendly pool intact and add
            # length-diverse samples to RL (50/50 256 vs [32,768]) plus
            # dedicated test buckets at [257,768] / [769,1536] / [1537,4096].
            # Runs post-TSEvol; extension samples are NOT paraphrased
            # (no LLM cost). Controlled by config.length_extension.*.
            # See memory/pipeline_length_generalization_2026_04_21.md.
            # ----------------------------------------------------------
            lx_cfg = getattr(config, "length_extension", {}) or {}
            if lx_cfg.get("enabled", False):
                from synth.align.length_extension import extend_with_lengths

                cfg_yaml = str(Path(sys.argv[0]).parent / "config" / "datagen_config.yaml")
                # Fallback: use a hard-coded default resolved from config path.
                if not Path(cfg_yaml).exists():
                    cfg_yaml = "config/datagen_config.yaml"

                targets = lx_cfg.get("targets", {}) or {}
                rl_n     = int(targets.get("train_rl", 0))
                val_n    = int(targets.get("val", 0))
                near_n   = int(targets.get("test_len_near", 0))
                far_n    = int(targets.get("test_len_far", 0))
                ext_n    = int(targets.get("test_len_extreme", 0))

                print("\n" + "=" * 60)
                print("LENGTH-EXTENSION PHASE (Option C curriculum)")
                print(f"  train_rl +{rl_n}  val +{val_n}  "
                      f"near={near_n}  far={far_n}  extreme={ext_n}")
                print("=" * 60)

                if rl_n > 0:
                    extras = extend_with_lengths(
                        config_yaml=cfg_yaml, split_label="train_rl",
                        target_count=rl_n, min_len=32, max_len=768,
                        default_prob=0.0, qa_filter="rl_only",
                    )
                    extras = filter_by_token_length(
                        extras, train_tokenizer, config.max_token_length,
                        stage="length-ext:rl",
                    )
                    rl_data.extend(extras)

                if val_n > 0:
                    extras = extend_with_lengths(
                        config_yaml=cfg_yaml, split_label="val",
                        target_count=val_n, min_len=32, max_len=768,
                        default_prob=0.0, qa_filter="none",
                    )
                    extras = filter_by_token_length(
                        extras, train_tokenizer, config.max_token_length,
                        stage="length-ext:val",
                    )
                    val_data.extend(extras)

                # Helper: length-extended samples PLUS OOD probes derived
                # from them. Running `generate_ood_for_split` on each bucket
                # gives us compositional probes at every length, enabling
                # the paper to report in-distribution vs OOD-length
                # compositional-transfer numbers separately.
                from synth.align.generators.ood_eval import generate_ood_for_split

                def _extend_and_ood(label, target_n, min_len, max_len,
                                    max_channels=0):
                    data = extend_with_lengths(
                        config_yaml=cfg_yaml, split_label=label,
                        target_count=target_n, min_len=min_len, max_len=max_len,
                        default_prob=0.0, qa_filter="none",
                        max_channels=max_channels,
                    )
                    data = filter_by_token_length(
                        data, train_tokenizer, config.max_token_length,
                        stage=f"length-ext:{label}",
                    )
                    # Run OOD generators on the fresh length-extended
                    # samples so compositional probes exist at this
                    # bucket. `max_ood` is the global cap; the stratified
                    # cap inside generate_ood_for_split splits it evenly
                    # per eval_type.
                    ood = generate_ood_for_split(data, max_ood_samples=max_ood)
                    print(f"  - {label}: {len(data)} base + {len(ood)} OOD")
                    data.extend(ood)
                    write_split_files(config, label, data)

                if near_n > 0:
                    _extend_and_ood("test_len_near", near_n, 257, 768)
                if far_n > 0:
                    _extend_and_ood("test_len_far", far_n, 769, 1536,
                                    max_channels=8)
                if ext_n > 0:
                    _extend_and_ood("test_len_extreme", ext_n, 1537, 4096,
                                    max_channels=8)

                print(f"\n  - Train RL (after ext):  {len(rl_data)}")
                print(f"  - Val      (after ext):  {len(val_data)}")
            # ----------------------------------------------------------

            write_split_files(config, "train_sft", sft_data)
            write_split_files(config, "train_rl", rl_data)
            write_split_files(config, "val", val_data)
            write_split_files(config, "test", test_data)
    finally:
        if llm_client is not None:
            llm_client.kill()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()