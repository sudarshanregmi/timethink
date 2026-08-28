import numpy as np

from typing import Dict, List, Tuple


tsevol_constraints = """- All details in the generated Q&A must be NEEDED to answer your question. Not just sourced from the Analysis Context — actually REQUIRED. Every metric, computation, and verdict must be necessary. If any line of the analysis could be removed and the answer still works, your question is too narrow.
- No generic statements. The answer must cite specific values, metrics, and computations from the Analysis Context.
- Use the EXACT metric names from the Analysis Context. Do NOT rename, substitute, or invent metric names (e.g., if the context says "Temperature", do not call it "Reactor Reading" or "Sensor A"). The question and answer must reference the same metric names that appear in the Analysis Context.
- No time series details in questions: use general language without mentioning specific numeric attributes (e.g., avoid "noise of 0.5" or "spike near position 100"). Specific details can only appear in the answer.
- When the question involves a comparison or threshold, include step-by-step reasoning in the answer (e.g., "mean=142.67, threshold=150, 142.67 < 150 → below threshold").
- One question, one answer. Keep it clear and concise.
- Output Format: Respond in JSON only: {"strategy": "strategy name", "question": "your question", "answer": "your answer"}. Do not include task labels like '#Given Q&A#' or '#Generated Q&A#'."""


comparison_instruction = """Here are two Q&A pairs and the Analysis Context they must be grounded in.

#Analysis Context#
{think_block}

#First Q&A#
{first_qa}

#Second Q&A#
{second_qa}

Are these two Q&A pairs equal? They are equal if ANY of the following hold:
    1. Their questions and answers are almost the same, with only minor word reordering.
    2. The second QA is a simple and obvious inference from the first QA.
    3. No difference in breadth or depth between the two QAs.

If equal, answer Equal.
If not equal, check ALL of these requirements:
    1. GROUNDING: All information in the second Q&A can be sourced from the Analysis Context and not generated without it.
    2. NO LEAKAGE: The question does not reveal specific numeric attributes (e.g., "noise of 0.5" or "spike near position 100") — those belong only in the answer.
    3. COMPLETENESS: Does the question REQUIRE all the information in the Analysis Context? If any metric, computation, or verdict could be removed and the answer still works → Invalid.
    4. FACTUAL ACCURACY: The second Q&A's answer must be factually correct according to the Analysis Context:
        a. Every number cited in the answer must match the corresponding value in the Analysis Context (minor rounding like 0.43 vs 0.4 is acceptable).
        b. Every comparison claim (greater/less, above/below, exceeds/within) must agree with the actual values in the Analysis Context.
        c. The final conclusion or verdict must logically follow from the evidence cited in the answer.
        If any number is wrong, any comparison is backwards, or the conclusion contradicts the cited evidence → Invalid.

If all requirements are met, answer Valid. Otherwise answer Invalid.

Your Judgement (Just answer: Equal/Invalid/Valid. No need to explain the reason.):"""


STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "Situation": "Create a virtual real-world scenario (industry, system, environment) that requires ALL the analysis information to answer. Best when Analysis Context is rich (many metrics/computations).",
    "Constraints": "Add one or more constraints/requirements to the seed Q&A, so the answer requires ALL the analysis to verify.",
    "Deepen": "Increase the depth and breadth of the seed Q&A so that ALL analysis information is needed.",
    "Concretize": "Replace general concepts with specific ones, making the question demand ALL the analysis details.",
    "ComplexReasoning": "Rewrite as a multi-step reasoning problem that requires ALL computations in the analysis.",
    "DeductiveReasoning": "Create a Yes/No question with a condition that requires ALL analysis information to evaluate.",
    "CausalReasoning": "Create a causal/effect question (multiple-choice) that requires ALL the analysis to reason about.",
}


def build_strategy_catalog(strategies: List[str]) -> str:
    """Build numbered strategy catalog from a list of strategy names."""
    return "\n".join(
        f"{i}. {name} — {STRATEGY_DESCRIPTIONS[name]}"
        for i, name in enumerate(strategies, 1)
    )


STRATEGY_CATALOG = build_strategy_catalog(list(STRATEGY_DESCRIPTIONS.keys()))

UNIFIED_EVOL_PROMPT = """=== PRIMARY DIRECTIVE ===
The evolved question MUST require the ENTIRE Analysis Context.
Every line, every metric, every computation, every verdict must be NECESSARY to answer.
If ANY line of the Analysis Context could be removed and the answer still works → INVALID.
=== END DIRECTIVE ===

#Analysis Context#
The following is the internal reasoning analysis of the time series data.
Your generated Q&A must require ALL of this analysis to answer.
{think_block}

#Available Strategies#
Choose the strategy that best fits this Analysis Context. Pick the one that will produce a question requiring ALL the information above.
{strategy_catalog}
{strategy_hint}
#Seed Q&A#
Use this as inspiration. Your evolved Q&A should be more complex/nuanced.
{seed_qa}

#Constraints#
{constraints}

Choose ONE strategy from the list above. Output JSON only: {{"strategy": "strategy name", "question": "your question", "answer": "your answer"}}"""


ALL_STRATEGY_NAMES = [
    "Situation", "Constraints", "Deepen", "Concretize",
    "ComplexReasoning", "DeductiveReasoning", "CausalReasoning",
]


# ---------------------------------------------------------------------------
# Paraphrase-only mode (verifiable-reward safe)
# ---------------------------------------------------------------------------
# Purpose: produce a new phrasing of the SEED question that asks the exact
# same thing. The seed's rule-based answer stays intact, so the RL reward
# signal remains verifiable and cannot be poisoned by LLM-generated labels.
#
# Used primarily for the RL split (0% full evolution, 100% paraphrase).
# The evolved sample keeps the ORIGINAL eval_type so it routes to the
# correct rule-based scorer in reward/__init__.py.

PARAPHRASE_PROMPT = """You will rewrite a time-series analysis question in different words. The rewritten question MUST ask for exactly the same thing as the original — same computation, same target, same answer.

#Seed Question#
{seed_question}

#What the question is actually asking for (do NOT change this)#
{think_block}

#Rules#
1. SAME MEANING: the paraphrased question must have the SAME correct answer as the seed. Do not add, remove, or alter any condition, metric, threshold, index, or quantity.
2. USE THE EXACT METRIC NAMES from the seed — do not rename, substitute, or invent metric names.
3. NO NUMERIC LEAKAGE: if the seed doesn't mention a specific numeric attribute (e.g., "noise of 0.5", "mean=142.67"), neither should your paraphrase.
4. NATURAL LANGUAGE: vary the sentence structure, word choice, and voice. You may change from interrogative to imperative ("What is X?" ↔ "Compute X.") and vice versa.
5. KEEP IT CONCISE: one question, no preamble, no explanation.
6. DO NOT generate an answer. Only rewrite the question.

Output JSON only: {{"question": "your rewritten question"}}"""


def createParaphrasePrompt(seed_question: str, think_block: str) -> str:
    return PARAPHRASE_PROMPT.format(
        seed_question=seed_question, think_block=think_block
    )


def createComparisonEliminatorPrompt(think_block, before, after):
    return comparison_instruction.format(
        think_block=think_block, first_qa=before, second_qa=after
    )


class EvolPrompt:
    def __init__(self, ts_idx: int, seed_q: str, seed_a: str, seed_fields: Dict[str, List[int]], instruction: str, timeseries: np.ndarray, attribute_pool: List[dict], metrics: List[str], corr_pool: List[Tuple[List[int], str]]):
        self.ts_idx = ts_idx
        self.timeseries = timeseries
        self.instruction = instruction

        self.all_fields = {"trend": range(len(timeseries)), "seasonal": range(len(timeseries)), "noise": range(len(timeseries)), "local": range(len(timeseries)), "statistic": range(len(timeseries)), "correlation": range(len(corr_pool))}
        self.fields = seed_fields
        self.qa_history = [(seed_q, seed_a)]

    def evol(self):
        # Get diff between fields and seed_fields
        diff_fields = {}
        for field in self.all_fields:
            if field not in self.fields:
                if len(self.all_fields[field]) > 0:
                    diff_fields[field] = self.all_fields[field]
            elif len(set(self.all_fields[field]) - set(self.fields[field])) > 0:
                diff_fields[field] = list(set(self.all_fields[field]) - set(self.fields[field]))
        
        # Random choose a field not in self.fields and add it
        if len(diff_fields) > 0:
            field = np.random.choice(list(diff_fields.keys()))
            self.fields.setdefault(field, [])
            self.fields[field].append(np.random.choice(diff_fields[field]))
    
    def push(self, q: str, a: str):
        self.qa_history.append((q, a))
        if len(self.qa_history) > 2:
            self.qa_history.pop(0)

    def generate_prompt(self):
        raise NotImplementedError("Subclasses must override generate_prompt()")

    def generate_comparison_prompt(self, q: str, a: str):
        raise NotImplementedError("Subclasses must override generate_comparison_prompt()")

    def to_dataset(self):
        return {
            "input": self.instruction + ' ' + self.qa_history[-1][0],
            "output": self.qa_history[-1][1],
            "timeseries": self.timeseries.tolist() if isinstance(self.timeseries, np.ndarray) else self.timeseries,
            "ts_idx": self.ts_idx,
            "fields": dict(sorted(self.fields.items()))
        }