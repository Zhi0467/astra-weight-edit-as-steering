from __future__ import annotations

from .prompt_suffix import append_problem_statement_suffix
from .prompt_format import format_user_prompt


def build_math_prompt(
    tokenizer,
    question: str,
    *,
    prompt_format: str = "auto",
    thinking_mode: str = "default",
    problem_statement_suffix: str | None = None,
) -> str:
    question = append_problem_statement_suffix(question, problem_statement_suffix)
    user_msg = f"{question}\n\nYou must put your final answer within \\boxed{{}}."
    return format_user_prompt(
        tokenizer,
        user_msg,
        prompt_format=prompt_format,
        thinking_mode=thinking_mode,
    )
