"""Загрузка промптов или их создание в langfuse.
Загрузка датасета из Langfuse."""

from typing import Literal
from langfuse import Langfuse
from langfuse.model import PromptClient

from settings import Settings
from constants import REFINE_PROMPT, GENERATION_PROMPT, ActInfo, RefineResponse


def _get_prompt_client(
    langfuse_client: Langfuse, prompt_name: str, prompt_version: Literal["best", "latest"] | int = "best"
) -> PromptClient:
    if isinstance(prompt_version, str):
        return langfuse_client.get_prompt(name=prompt_name, label=prompt_version)
    return langfuse_client.get_prompt(name=prompt_name, version=prompt_version)


def _create_prompt_client(
    langfuse_client: Langfuse, prompt_name: str, prompt: str, config: dict | None = None
) -> PromptClient:
    return langfuse_client.create_prompt(name=prompt_name, prompt=prompt, type="text", config=config)


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
        prompt_client = _create_prompt_client(
            langfuse_client,
            prompt_name,
            GENERATION_PROMPT,
            {"json_schema": ActInfo.model_json_schema()},
        )
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
        prompt_client = _create_prompt_client(
            langfuse_client,
            prompt_name,
            REFINE_PROMPT,
            {"json_schema": RefineResponse.model_json_schema()},
        )
    return prompt_client


def get_train_test_dataset(langfuse_client: Langfuse, settings: Settings):
    train_dataset = langfuse_client.get_dataset(settings.LANGFUSE_TRAIN_DATASET_NAME)
    test_dataset = langfuse_client.get_dataset(settings.LANGFUSE_TEST_DATASET_NAME)
    return train_dataset, test_dataset
