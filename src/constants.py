from typing import Annotated, Any, TypedDict

from langfuse.model import PromptClient
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field


class ActInfo(BaseModel):
    full_name: str | None = Field(
        default=None, description="Полное наименование правового акта (тип + наименование государственного органа)"
    )
    publication_date: str | None = Field(default=None, description="Дата опубликования документа")
    number: str | None = Field(default=None, description="Номер документа (например, 3789-р)")
    title: str | None = Field(default=None, description="Заголовок документа")
    government_agency_name: str | None = Field(default=None, description="Наименование государственного органа")
    signatory: str | None = Field(default=None, description="Официальное лицо, подписавшее акт")
    type: str | None = Field(default=None, description="Тип правового акта")


class BadResult(TypedDict):
    text: str
    reference_answers: dict[str, Any]
    generated_answers: dict[str, Any]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    repeat: int
    current_step: int
    bad_results: list[BadResult]
    best_test_accuracy: float
    is_test_accuracy_improving: bool
    best_generation_prompt_version: int
    generation_prompt_client: PromptClient
    refine_prompt_client: PromptClient


class RefineResponse(BaseModel):
    changes: str = Field(description="Какие были внесены изменения и почему")
    new_prompt: str = Field(description="Улучшенная версия промпта")


GENERATION_PROMPT = "Извлеки данные из документа согласно заданной схеме.\n# Документ\n{{document}}\n"

REFINE_PROMPT = (
    "Ты анализируешь работу маленькой модели для извлечения данных из документов. "
    "Тебе дан промпт модели, схема выходных данных и примеры документов "
    "с указанием, где модель ошиблась. Проанализируй ошибки и дополни "
    "или полностью перепиши промпт для извлечения данных любыми известными "
    "техниками, например, задание роли, добавление примеров и ограничений "
    "и другое. В ответе верни улучшенную версию промпта и какие изменения ты внёс.\n"
    "# Промпт модели\n{{generation_prompt}}\n\n"
    "# Схема выходных данных модели\n{{json_schema}}\n\n"
    "# Примеры данных\n{{examples}}\n"
)
