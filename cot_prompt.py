from __future__ import annotations


OLD_PROMPT_PREFIX = "Problem:\n"
OLD_REFERENCE_SEPARATOR = "\n\nFinal output:\n"
OLD_PROMPT_SUFFIX = "\n\nWrite the reasoning chain that leads to this final output."


def build_cot_reconstruction_prompt(question: str, reference_answer: str) -> str:
    """Build the shared prompt used for DPO and synthetic CoT generation."""
    return (
        "Below are a question and its reference answer. Develop a\n"
        "**clear, step-by-step line of reasoning** that arrives at the\n"
        "reference answer, expressed as a **first-person think-aloud\n"
        "monologue**.\n\n"
        "Use the reference answer **privately** to guide the direction\n"
        "and organization of the reasoning. The resulting monologue\n"
        "should nevertheless sound like an **authentic solution being\n"
        "worked out in real time on a first encounter**, with **no mention\n"
        "or suggestion that the reference answer was available in\n"
        "advance**.\n\n"
        f"Question: {question.strip()}\n"
        f"Reference Answer: {reference_answer.strip()}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "1. Return **ONLY the first-person think-aloud monologue**.\n"
        "2. Write in the voice of a **focused person reasoning\n"
        "   privately to themself**.\n"
        "3. **Never state or imply** that you previously saw or used the\n"
        "   Reference Answer."
    )


def upgrade_legacy_cot_prompt(prompt: str) -> str:
    """Convert this project's legacy DPO prompt while leaving other text intact."""
    stripped = prompt.strip()
    if not stripped.startswith(OLD_PROMPT_PREFIX) or not stripped.endswith(
        OLD_PROMPT_SUFFIX
    ):
        return stripped

    body = stripped[len(OLD_PROMPT_PREFIX) : -len(OLD_PROMPT_SUFFIX)]
    if OLD_REFERENCE_SEPARATOR not in body:
        return stripped
    question, reference_answer = body.split(OLD_REFERENCE_SEPARATOR, 1)
    if not question.strip() or not reference_answer.strip():
        return stripped
    return build_cot_reconstruction_prompt(question, reference_answer)
