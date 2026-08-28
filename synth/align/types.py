from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class GenerationResult:
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    llm_prompts: List[List[str]] = field(default_factory=list)
    fields: List[Dict] = field(default_factory=list)
    qa_types: List[str] = field(default_factory=list)
    eval_tasks: List[str] = field(default_factory=list)
    eval_metadatas: List[Dict] = field(default_factory=list)
    correlations: List[Dict] = field(default_factory=list)
    clusters: List[Dict] = field(default_factory=list)
    corr_pool_list: List[Any] = field(default_factory=list)