import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
import torch

load_dotenv()


def get_openrouter_client():
    headers = {}
    if os.getenv("HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("HTTP_REFERER")
    if os.getenv("X_TITLE"):
        headers["X-Title"] = os.getenv("X_TITLE")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers=headers if headers else None,
    )


def get_local_client(port: int = 3002):
    return OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


if not torch.cuda.is_available:
    client = get_openrouter_client()
    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL", "alibaba/tongyi-deepresearch-30b-a3b:free"
    )
else:
    client = get_local_client()
    OPENROUTER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def get_llm_response(
    prompt: str, system_prompt: str = "You are a world-class research assistant."
) -> str:
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    content = content if content else ""
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
