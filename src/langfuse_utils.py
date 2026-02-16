"""Загрузка промптов или их создание в langfuse.
Загрузка датасета из Langfuse."""
import asyncio
from typing import Literal
from langfuse import Langfuse
from langfuse.model import PromptClient
from loguru import logger

from src.settings import Settings
from src.constants import REFINE_PROMPT, GENERATION_PROMPT, ActInfo, RefineResponse

def _get_prompt_client(
    langfuse_client: Langfuse, prompt_name: str, prompt_version: Literal["best", "latest"] | int = "best"
) -> PromptClient:
    if isinstance(prompt_version, str):
        return langfuse_client.get_prompt(name=prompt_name, label=prompt_version)
    return langfuse_client.get_prompt(name=prompt_name, version=prompt_version)


def _create_prompt_client(
    langfuse_client: Langfuse,
    prompt_name: str,
    prompt: str,
    label: str,
    config: dict | None = None,
) -> PromptClient:
    return langfuse_client.create_prompt(name=prompt_name, prompt=prompt, labels=[label], type="text", config=config)


def get_generation_prompt_client(langfuse_client: Langfuse, settings: Settings) -> PromptClient:
    """Создаёт или загружает промпт для маленькой модели из Langfuse

    Args:
        settings (Settings): настройки программы с указаниями версии и имени промпта

    Returns:
        PromptClient: клиент для промпта
    """
    prompt_name = settings.LANGFUSE_GENERATION_PROMPT_NAME
    prompt_version = settings.LANGFUSE_GENERATION_PROMPT_VERSION
    try:
        prompt_client = _get_prompt_client(langfuse_client, prompt_name, prompt_version)
    except:
        logger.info(f"Промпт с названием '{prompt_name}' и версией '{prompt_version}' не существует. Он будет создан.")
        prompt_client = _create_prompt_client(
            langfuse_client,
            prompt_name,
            GENERATION_PROMPT,
            "best",
            {"json_schema": ActInfo.model_json_schema()},
        )
    logger.info(f"Промпт для маленькой модели {prompt_client.name} v{prompt_client.version}")
    return prompt_client


def get_refine_prompt_client(langfuse_client: Langfuse, settings: Settings) -> PromptClient:
    """Создаёт или загружает промпт для большой модели из Langfuse

    Args:
        settings (Settings): настройки программы с указаниями версии и имени промпта

    Returns:
        PromptClient: клиент для промпта
    """
    prompt_name = settings.LANGFUSE_REFINE_PROMPT_NAME
    prompt_version = settings.LANGFUSE_REFINE_PROMPT_VERSION
    try:
        prompt_client = _get_prompt_client(langfuse_client, prompt_name, prompt_version)
    except:
        logger.info(f"Промпт с названием '{prompt_name}' и версией '{prompt_version}' не существует. Он будет создан.")
        prompt_client = _create_prompt_client(
            langfuse_client,
            prompt_name,
            REFINE_PROMPT,
            "best",
            {"json_schema": RefineResponse.model_json_schema()},
        )
    logger.info(f"Промпт для большой модели {prompt_client.name} v{prompt_client.version}")
    return prompt_client


def get_train_test_dataset(langfuse_client: Langfuse, settings: Settings):
    train_dataset = langfuse_client.get_dataset(settings.LANGFUSE_TRAIN_DATASET_NAME)
    test_dataset = langfuse_client.get_dataset(settings.LANGFUSE_TEST_DATASET_NAME)
    logger.info("train и test datasets загружены")
    return train_dataset, test_dataset


async def get_run_scores(langfuse_client: Langfuse, dataset_name: str, run_name: str):
    run = await langfuse_client.async_api.datasets.get_run(dataset_name, run_name)

    tasks = []
    for item in run.dataset_run_items:
        tasks.append(langfuse_client.async_api.trace.get(item.trace_id))

    run_scores = []
    traces = await asyncio.gather(*tasks)
    for trace in traces:
        score = trace.scores[0].value
        run_scores.append(score)

    return sum(run_scores) / len(run_scores), trace.observations[0].prompt_version


async def get_dataset_scores(langfuse_client: Langfuse, dataset_name: str):
    runs = await langfuse_client.async_api.datasets.get_runs(dataset_name)
    scores = []

    tasks = [get_run_scores(langfuse_client, dataset_name, run.name) for run in runs.data]
    results = await asyncio.gather(*tasks)
    for run, (score, prompt_version) in zip(runs.data, results):
        scores.append(
            {
                "run_name": run.name,
                "score": score,
                "prompt_version": prompt_version,
            }
        )
    return scores

async def get_best_generation_prompt_version(langfuse_client: Langfuse, dataset_name: str) -> tuple[int, float]:
    scores = await get_dataset_scores(langfuse_client, dataset_name)
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    logger.info(f"Best generation prompt is {scores[0]}")
    return int(scores[0]["prompt_version"]), scores[0]["score"]