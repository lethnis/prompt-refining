import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.constants import ActInfo, RefineResponse

load_dotenv()

refine_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model_name=os.getenv("OPENAI_REFINE_MODEL"),
    timeout=60,
    max_retries=3,
).with_structured_output(RefineResponse, include_raw=True)

generation_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model_name=os.getenv("OPENAI_GENERATION_MODEL"),
    timeout=60,
    max_retries=3,
).with_structured_output(ActInfo)
