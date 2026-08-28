import re
import typing as t


def clean_json_text(json_str: str) -> t.Optional[str]:
    if not json_str or not isinstance(json_str, str):
        return None

    target = None

    code_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", json_str, re.IGNORECASE)
    if code_blocks:
        target = code_blocks[-1].strip()
    else:
        candidate_blocks = []
        brace_level = 0
        in_string = False
        is_escaped = False
        start_idx = -1

        for i, char in enumerate(json_str):
            if in_string:
                if char == '\\': is_escaped = not is_escaped
                elif char == '"' and not is_escaped: in_string = False
                else: is_escaped = False
            else:
                if char == '"': in_string = True
                elif char == '{':
                    if brace_level == 0: start_idx = i
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0 and start_idx != -1:
                        candidate_blocks.append(json_str[start_idx : i + 1])
                        start_idx = -1

        if candidate_blocks:
            target = candidate_blocks[-1].strip()

    if not target:
        return None

    target = re.sub(r'(?<!\\)(?<![\{\[\s:])\"(?![:,\s\}\]])', r'\"', target)
    target = re.sub(r':\s*True\b', ': true', target)
    target = re.sub(r':\s*False\b', ': false', target)
    target = re.sub(r':\s*None\b', ': null', target)
    target = re.sub(r',\s*([\]}])', r'\1', target)
    return target


def strip_think_block(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_subject_from_question(q_text):
    match = re.search(r"related to (time series \d+)", q_text)
    if match:
        return match.group(1)

    match = re.search(r"between (.*?) and (.*?)(\?|$)", q_text)
    if match:
        return f"{match.group(1)}, {match.group(2)}"

    return "the subject mentioned in the question"


