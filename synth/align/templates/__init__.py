"""Prompt template registry and text augmentation for diverse QA generation."""
from synth.align.templates.prompts import PromptRegistry
from synth.align.templates.perturbations import TextAugmenter

__all__ = ["PromptRegistry", "TextAugmenter"]
