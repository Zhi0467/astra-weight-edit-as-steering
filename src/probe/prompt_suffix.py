from __future__ import annotations


def append_problem_statement_suffix(prompt: str, suffix: str | None) -> str:
    suffix_text = (suffix or "").strip()
    if not suffix_text:
        return prompt.rstrip()
    return f"{prompt.rstrip()}\n\n{suffix_text}"
