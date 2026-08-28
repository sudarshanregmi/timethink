"""Functional tests for QA generators — verify structural correctness of generated output."""
import pytest
import re
from synth.align.config import Config, Mode, Difficulty, PromptIndexer
from synth.align.factory import generate_sample
from synth.ts_generator.utils.common_utils import load_cfg
from synth.ts_generator.utils.probability_utils import DEFAULT_QA_TYPE_WEIGHTS


# Eval types that may produce data-only think blocks (no ===) for lookup sub-types
_DATA_ONLY_POSSIBLE_TYPES = {
    'description', 'stat_numerical', 'periodicity',
}

# Taxonomy types use free-form think blocks (no === required).
# The model reasons however it wants for these types.
_TAXONOMY_PREFIXES = ('atomic_', 'rl_', 'ood_', 'bridge_')


def _needs_separator(eval_type: str) -> bool:
    """Return True if this eval_type requires === in its think block."""
    if eval_type in _DATA_ONLY_POSSIBLE_TYPES:
        return False
    if eval_type.startswith(_TAXONOMY_PREFIXES):
        return False
    return True


@pytest.fixture(scope="module")
def config():
    metric_config = load_cfg("config/metric_set.json")
    return Config(
        num_data=10,
        encoding_method="no",
        output_base_dir="/tmp/test_qa",
        dryrun=True,
        debug=True,
        local_llm_path="",
        disable_metric_config=False,
        metric_config=metric_config,
        qa_type_weights=DEFAULT_QA_TYPE_WEIGHTS,
    )


def _generate_debug_sample(config, mode):
    """Generate a debug sample (all QAs kept, no single-QA selection)."""
    indexer = PromptIndexer()
    return generate_sample(
        mode=mode, config=config, indexer=indexer,
        seq_len=256, difficulty=Difficulty.EASY, debug=True,
    )


class TestUTSGeneration:
    def test_generates_qa(self, config):
        result = _generate_debug_sample(config, Mode.UTS)
        assert len(result.questions) > 0, "UTS should generate at least 1 QA"

    def test_all_lists_same_length(self, config):
        result = _generate_debug_sample(config, Mode.UTS)
        n = len(result.questions)
        assert len(result.answers) == n
        assert len(result.qa_types) == n
        assert len(result.eval_tasks) == n
        assert len(result.eval_metadatas) == n

    def test_all_eval_metadatas_have_length(self, config):
        result = _generate_debug_sample(config, Mode.UTS)
        for i, meta in enumerate(result.eval_metadatas):
            assert 'length' in meta, f"eval_metadatas[{i}] missing 'length' key"

    def test_answers_have_think_block(self, config):
        result = _generate_debug_sample(config, Mode.UTS)
        for i, answer in enumerate(result.answers):
            assert '<think>' in answer, f"answers[{i}] missing <think> tag"
            assert '</think>' in answer, f"answers[{i}] missing </think> tag"

    def test_think_block_has_separator(self, config):
        result = _generate_debug_sample(config, Mode.UTS)
        for i, answer in enumerate(result.answers):
            think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                # Data-only blocks (lookups + description) have no ===
                if _needs_separator(result.eval_tasks[i]):
                    assert '===' in think_content, (
                        f"answers[{i}] (eval_type={result.eval_tasks[i]}) "
                        f"missing === separator in think block"
                    )

    def test_eval_tasks_are_known_types(self, config):
        from evaluation.eval.config import SEGMENT_FAMILY_TYPES
        known = SEGMENT_FAMILY_TYPES | {'description', 'correlation', 'anticorrelation',
                                         'clustering', 'anticlustering', 'yes_no'}
        result = _generate_debug_sample(config, Mode.UTS)
        for i, task in enumerate(result.eval_tasks):
            assert task in known or task.startswith(_TAXONOMY_PREFIXES), (
                    f"eval_tasks[{i}] = {task!r} is not a known eval_type"
                )


class TestMTSShapeGeneration:
    def test_generates_qa(self, config):
        result = _generate_debug_sample(config, Mode.MTS_SHAPE)
        assert len(result.questions) > 0, "MTS_SHAPE should generate at least 1 QA"

    def test_all_lists_same_length(self, config):
        result = _generate_debug_sample(config, Mode.MTS_SHAPE)
        n = len(result.questions)
        assert len(result.answers) == n
        assert len(result.eval_metadatas) == n

    def test_all_eval_metadatas_have_length(self, config):
        result = _generate_debug_sample(config, Mode.MTS_SHAPE)
        for i, meta in enumerate(result.eval_metadatas):
            assert 'length' in meta, f"eval_metadatas[{i}] missing 'length' key"

    def test_answers_have_think_block(self, config):
        result = _generate_debug_sample(config, Mode.MTS_SHAPE)
        for i, answer in enumerate(result.answers):
            assert '<think>' in answer, f"answers[{i}] missing <think> tag"
            assert '</think>' in answer, f"answers[{i}] missing </think> tag"

    def test_think_block_has_separator(self, config):
        result = _generate_debug_sample(config, Mode.MTS_SHAPE)
        for i, answer in enumerate(result.answers):
            think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                if _needs_separator(result.eval_tasks[i]):
                    assert '===' in think_content, (
                        f"answers[{i}] (eval_type={result.eval_tasks[i]}) "
                        f"missing === separator in think block"
                    )

    def test_eval_tasks_are_known_types(self, config):
        from evaluation.eval.config import SEGMENT_FAMILY_TYPES
        known = SEGMENT_FAMILY_TYPES | {'description', 'correlation', 'anticorrelation',
                                         'clustering', 'anticlustering', 'yes_no'}
        result = _generate_debug_sample(config, Mode.MTS_SHAPE)
        for i, task in enumerate(result.eval_tasks):
            assert task in known or task.startswith(_TAXONOMY_PREFIXES), (
                    f"eval_tasks[{i}] = {task!r} is not a known eval_type"
                )


class TestMTSLocalGeneration:
    def test_generates_qa(self, config):
        result = _generate_debug_sample(config, Mode.MTS_LOCAL)
        assert len(result.questions) > 0, "MTS_LOCAL should generate at least 1 QA"

    def test_all_lists_same_length(self, config):
        result = _generate_debug_sample(config, Mode.MTS_LOCAL)
        n = len(result.questions)
        assert len(result.answers) == n
        assert len(result.eval_metadatas) == n

    def test_all_eval_metadatas_have_length(self, config):
        result = _generate_debug_sample(config, Mode.MTS_LOCAL)
        for i, meta in enumerate(result.eval_metadatas):
            assert 'length' in meta, f"eval_metadatas[{i}] missing 'length' key"

    def test_answers_have_think_block(self, config):
        result = _generate_debug_sample(config, Mode.MTS_LOCAL)
        for i, answer in enumerate(result.answers):
            assert '<think>' in answer, f"answers[{i}] missing <think> tag"
            assert '</think>' in answer, f"answers[{i}] missing </think> tag"

    def test_think_block_has_separator(self, config):
        result = _generate_debug_sample(config, Mode.MTS_LOCAL)
        for i, answer in enumerate(result.answers):
            think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
                if _needs_separator(result.eval_tasks[i]):
                    assert '===' in think_content, (
                        f"answers[{i}] (eval_type={result.eval_tasks[i]}) "
                        f"missing === separator in think block"
                    )

    def test_eval_tasks_are_known_types(self, config):
        from evaluation.eval.config import SEGMENT_FAMILY_TYPES
        known = SEGMENT_FAMILY_TYPES | {'description', 'correlation', 'anticorrelation',
                                         'clustering', 'anticlustering', 'yes_no'}
        result = _generate_debug_sample(config, Mode.MTS_LOCAL)
        for i, task in enumerate(result.eval_tasks):
            assert task in known or task.startswith(_TAXONOMY_PREFIXES), (
                    f"eval_tasks[{i}] = {task!r} is not a known eval_type"
                )


class TestSingleQASelection:
    """Test that select_single_qa produces valid single-QA output."""
    def test_uts_single_qa(self, config):
        non_debug_config = Config(
            num_data=1, encoding_method="no", output_base_dir="/tmp/test",
            dryrun=True, debug=False, local_llm_path="",
            disable_metric_config=False, metric_config=config.metric_config,
            qa_type_weights=DEFAULT_QA_TYPE_WEIGHTS,
        )
        indexer = PromptIndexer()
        result = generate_sample(
            mode=Mode.UTS, config=non_debug_config, indexer=indexer,
            seq_len=256, difficulty=Difficulty.EASY,
        )
        assert len(result.questions) == 1, "Non-debug should produce exactly 1 QA"
        assert len(result.answers) == 1
        assert len(result.eval_metadatas) == 1
        assert 'length' in result.eval_metadatas[0]
