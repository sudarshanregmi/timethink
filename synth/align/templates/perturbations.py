"""Text augmentation utilities for instruction diversity."""

import random
import string


# Filler phrases to optionally prepend
_FILLERS = [
    "Um, ",
    "So, ",
    "Basically, ",
    "Hey, ",
    "Well, ",
    "Alright, ",
    "OK so ",
    "Let me ask: ",
    "Quick question — ",
    "I'm curious, ",
]

# Technical → casual phrasing swaps
_STYLE_MAP = {
    "analyze": "look at",
    "characteristics": "behavior",
    "fluctuation": "spike or dip",
    "correlation": "connection",
    "periodicity": "repeating pattern",
    "monotonic tendency": "general direction",
    "perspectives": "angles",
    "significant": "noticeable",
    "local event": "local",
    "conclude the physical meaning": "explain what this means physically",
    "in detail": "closely",
}


class TextAugmenter:
    """Applies lightweight perturbations to instruction text for diversity."""

    @staticmethod
    def add_typos(text: str, prob: float = 0.08) -> str:
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and random.random() < prob:
                # Pick a random operation
                op = random.choice(["swap", "drop"])
                chars = list(word)
                if op == "swap" and len(chars) > 2:
                    idx = random.randint(1, len(chars) - 2)
                    chars[idx], chars[idx - 1] = chars[idx - 1], chars[idx]
                elif op == "drop":
                    idx = random.randint(1, len(chars) - 1)
                    chars.pop(idx)
                result.append("".join(chars))
            else:
                result.append(word)
        return " ".join(result)

    @staticmethod
    def add_fillers(text: str, prob: float = 0.15) -> str:
        if random.random() < prob:
            filler = random.choice(_FILLERS)
            # Lowercase the first character of the original text after filler
            if text and text[0].isupper():
                text = text[0].lower() + text[1:]
            return filler + text
        return text

    @staticmethod
    def apply_style(text: str, style: str = "casual") -> str:
        if style != "casual":
            return text
        result = text
        for formal, casual in _STYLE_MAP.items():
            result = result.replace(formal, casual)
        return result

    @classmethod
    def augment(cls, text: str, prob: float = 0.2) -> str:
        if random.random() > prob:
            return text

        op = random.choice(["typos", "fillers", "style"])
        if op == "typos":
            return cls.add_typos(text)
        elif op == "fillers":
            return cls.add_fillers(text, prob=1.0)  # Force filler since we already rolled
        else:
            return cls.apply_style(text, style="casual")
