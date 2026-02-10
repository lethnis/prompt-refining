import os
from datetime import date

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model_name=os.getenv("OPENAI_REFINE_MODEL"),
    timeout=30,
    max_retries=3
)

structured_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model_name=os.getenv("OPENAI_STRUCTURED_MODEL"),
    timeout=30,
    max_retries=3
)

class ActInfo(BaseModel):
    full_name: str | None = Field(default=None, description="Полное наименование правового акта (тип + наименование государственного органа)")
    publication_date: date | None = Field(default=None, description="Дата опубликования документа")
    number: str | None = Field(default=None, description="Номер документа (например, 3789-р)")
    title: str | None = Field(default=None, description="Заголовок документа")
    government_agency_name: str | None = Field(default=None, description="Наименование государственного органа")
    signatory: str | None = Field(default=None, description="Официальное лицо, подписавшее акт")
    type: str | None = Field(default=None, description="Тип правового акта")