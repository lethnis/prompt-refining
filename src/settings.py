from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Настройки моделей и Langfuse для автоподбора промптов.

    Для переменных Langfuse:
    1) Зайти в настройки проекта слева снизу
    2) Выбрать вкладку API keys
    3) Создать новые ключи и скопировать их в .env"""

    OPENAI_BASE_URL: str = Field(description="URL до провайдера моделей")
    OPENAI_API_KEY: str = Field(description="Ключ доступа")
    OPENAI_REFINE_MODEL: str = Field(
        description="Имя большой модели, которая будет улучшать промпт. Модель должна поддерживать Structured Output"
    )
    OPENAI_GENERATION_MODEL: str = Field(
        description="Имя маленькой модели, которая решает конкретную задачу. Должна поддерживать Structured Output"
    )

    LANGFUSE_SECRET_KEY: str = Field(description="Секретный ключ доступа для Langfuse")
    LANGFUSE_PUBLIC_KEY: str = Field(description="Открытый ключ доступа для Langfuse")
    LANGFUSE_BASE_URL: str = Field(description="URL, где поднят Langfuse")

    LANGFUSE_GENERATION_PROMPT_NAME: str = Field(
        default="generation-prompt",
        description="Название промпта для маленькой модели",
    )
    LANGFUSE_GENERATION_PROMPT_VERSION: Literal["best", "latest"] | int = Field(
        default="best",
        description=(
            "Версия промпта для маленькой модели. Если промпта не существует,"
            "он будет создан из промпта 'GENERATION_PROMPT' в файле `constants.py`"
        ),
    )
    LANGFUSE_REFINE_PROMPT_NAME: str = Field(
        default="refine-prompt",
        description="Название промпта для большой модели",
    )
    LANGFUSE_REFINE_PROMPT_VERSION: Literal["best", "latest"] | int = Field(
        default="best",
        description=(
            "Версия промпта для большой модели. Если промпта не существует,"
            "он будет создан из промпта 'REFINE_PROMPT' в файле `constants.py`"
        ),
    )
    LANGFUSE_TRAIN_DATASET_NAME: str = Field(
        default="RusLawOD/train",
        description="Название датасета для обновления промпта",
    )
    LANGFUSE_TEST_DATASET_NAME: str = Field(
        default="RusLawOD/test",
        description="Название датасета для тестирования нового промпта",
    )
    DRAW_AGENT_GRAPH: str | None = Field(
        default=None,
        description="Куда сохранить .png рисунок агента",
        examples=["agent_graph.png"],
    )

    AGENT_REFINE_STEPS: int = Field(
        default=3,
        description="Сколько раз выполнить цикл улучшения промпта"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,
        cli_kebab_case=True,
        extra="ignore",
    )
