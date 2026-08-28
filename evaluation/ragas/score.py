from evaluation.ragas.metric import AnswerCorrectness
from evaluation.ragas.config import load_llm, load_embeddings, config
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig
from loguru import logger
import copy
import asyncio


def calculate_ragas_score(question: str, response: str, label: str):
    try:
        answer_correctness = AnswerCorrectness(
            embeddings=load_embeddings(),
            llm=load_llm(),
            weights=[1.0, 0.0]
        )
        
        answer_correctness.init(RunConfig())
        answer_correctness.answer_detail = {}

        sample = SingleTurnSample(
            user_input=question,
            response=response,
            reference=label
        )

        score_value = answer_correctness.single_turn_score(sample)
        
    except Exception as e:
        logger.error(f"Error calculating RAGAS score: {e}. Please make sure that you have set the correct API key!")
        raise

    return float(score_value), copy.deepcopy(answer_correctness.answer_detail)
