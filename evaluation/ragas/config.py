import os
import sys
from loguru import logger
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

CONFIG_PATH = os.getenv('CONFIG_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), './config/config.toml'))


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error(f'Config file does not exist: {CONFIG_PATH}')
        sys.exit(1)
    with open(CONFIG_PATH, 'rb') as f:
        cfg = tomllib.load(f)
    return cfg


config = load_config()


def load_llm() -> LangchainLLMWrapper:
    models_config = config.get('models')
    llm_type = models_config.get('llm_type', 'openai')
    model = None

    if llm_type == 'openai':
        api_base = models_config.get('openai_api_base', 'http://localhost:8000/v1')
        api_key = models_config.get('openai_api_key', 'EMPTY')
        model_name = models_config.get('llm_model', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4')
        temperature = models_config.get('temperature', 0.0)
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key
        from langchain_openai.chat_models import ChatOpenAI
        model = ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=temperature,
            max_retries=3,
            streaming=False
        )
    elif llm_type == 'tongyi':
        os.environ["DASHSCOPE_API_KEY"] = models_config.get('dashscope_api_key', '')
        from langchain_community.chat_models.tongyi import ChatTongyi
        model = ChatTongyi(model=models_config.get('llm_model', 'qwen1.5-72b-chat'))
    elif llm_type == 'glm':
        os.environ["OPENAI_API_BASE"] = models_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = models_config.get('openai_api_key', '')
        from langchain_community.chat_models import ChatZhipuAI
        model = ChatZhipuAI(
            temperature=models_config.get('temperature', 1),
            api_key=models_config.get('openai_api_key', ''),
            model=models_config.get('llm_model', 'gpt-3.5-turbo-16k')
        )

    if model:
        return LangchainLLMWrapper(model)

    logger.error(f'Unsupported LLM model: {llm_type}')
    sys.exit(1)


def load_embeddings() -> LangchainEmbeddingsWrapper:
    embedding_config = config.get('embedding')
    emb_type = embedding_config.get('emb_type', 'openai')
    embeddings = None
    if emb_type == 'openai':
        os.environ["OPENAI_API_BASE"] = embedding_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = embedding_config.get('openai_api_key', '')
        from langchain_openai.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=embedding_config.get('embeddings_model', 'text-embedding-ada-002'))
    elif emb_type == 'dashscope':
        os.environ["DASHSCOPE_API_KEY"] = embedding_config.get('dashscope_api_key', '')
        from langchain_community.embeddings.dashscope import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(model=embedding_config.get('embeddings_model', 'text-embedding-v2'))
    elif emb_type == 'huggingface':
        embeddings = HuggingFaceEmbeddings(model_name=embedding_config.get('embeddings_model', 'BAAI/bge-small-en-v1.5'))

    if embeddings:
        return LangchainEmbeddingsWrapper(embeddings)
    logger.error(f'Unsupported Embeddings model: {emb_type}')
    sys.exit(1)