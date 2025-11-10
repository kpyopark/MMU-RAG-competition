"""
LLM client wrapper using GeminiClient for all operations.

This module provides backward-compatible functions that now use the Gemini API
instead of vLLM/OpenRouter. The function signatures remain the same to avoid
breaking existing pipeline code.
"""

import os
from dotenv import load_dotenv
from loguru import logger

from .gemini_client import GeminiClient

load_dotenv()

# Initialize Gemini client (single source of truth)
client = GeminiClient(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
)


def get_llm_response(
    prompt: str, system_prompt: str = "You are a world-class research assistant."
) -> str:
    """
    Generate LLM response using Gemini API.

    This function maintains backward compatibility with the original OpenAI-style
    interface while using the new GeminiClient under the hood.

    Args:
        prompt: User prompt
        system_prompt: System instruction

    Returns:
        Generated text content
    """
    content = client.complete(
        prompt=prompt,
        system_prompt=system_prompt,
    )
    logger.debug(f"Prompt[:200]: {prompt[:200]}\nResponse[:200]: {content[:200]}")
    return content


def self_evolve(
    initial_prompt: str,
    system_prompt: str,
    num_variants: int,
    evolution_steps: int,
) -> tuple[str, list[str]]:
    """
    Implements the Component-wise Self-Evolution algorithm from the TTD-DR paper.
    It generates multiple variants, critiques and refines them, then merges them.
    """
    # 1. Initial States: Generate diverse variants
    variants = [
        get_llm_response(initial_prompt, system_prompt) for _ in range(num_variants)
    ]

    for i in range(evolution_steps):
        evolved_variants = []
        for variant in variants:
            critique_prompt = f"""
            Critique the following text based on the original request. Provide a concise critique and a fitness score from 1 to 10.
            Then, rewrite the text to address the critique.

            Original Request: {initial_prompt}

            Text to Critique:
            ---
            {variant}
            ---

            Provide your response in the following format, and nothing else:
            CRITIQUE: [Your critique here]
            SCORE: [Your score here]
            REVISED_TEXT: [Your improved version of the text]
            """

            feedback_response = get_llm_response(
                critique_prompt, "You are a critical and constructive reviewer."
            )

            if feedback_response is not None:
                # Simple parsing of the structured response
                revised_text = feedback_response.split("REVISED_TEXT:")[1].strip()
                evolved_variants.append(revised_text)
            else:
                # If parsing fails, just use the original variant
                evolved_variants.append(variant)
        variants = evolved_variants

    # 4. Cross-over (Merge)
    merge_prompt = f"""
    You are given several refined texts that all attempt to answer an original request.
    Synthesize them into a single, comprehensive, and superior final text.

    Original Request: {initial_prompt}

    Refined Texts to Merge:
    ---
    {"---".join(variants)}
    ---

    Produce the final, merged text.
    """
    final_merged_text = get_llm_response(merge_prompt, system_prompt)
    return final_merged_text, variants
