import asyncio
from datetime import datetime
import random
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from loguru import logger
from tqdm import tqdm

from src.models import refine_model, generation_model
from src.constants import BadResult, State
from src.langfuse_utils import get_generation_prompt_client, get_refine_prompt_client, get_train_test_dataset, get_best_generation_prompt_version
from src.settings import Settings

load_dotenv()

langfuse_client = get_client()
callback_handler = CallbackHandler()


def main():
    settings = Settings()

    test_dataset_name = quote(settings.LANGFUSE_TEST_DATASET_NAME, safe="")
    best_version, best_test_accuracy = asyncio.run(get_best_generation_prompt_version(langfuse_client, test_dataset_name))
    settings.LANGFUSE_GENERATION_PROMPT_VERSION = best_version

    generation_prompt_client = get_generation_prompt_client(langfuse_client, settings)
    refine_prompt_client = get_refine_prompt_client(langfuse_client, settings)

    train_dataset, test_dataset = get_train_test_dataset(langfuse_client, settings)

    logger.info("Агент создаётся")
    agent = build_agent()

    if settings.DRAW_AGENT_GRAPH is not None:
        try:
            agent.get_graph().draw_mermaid_png(output_file_path=settings.DRAW_AGENT_GRAPH, max_retries=3)
        except:
            logger.info("Не удалось сохранить изображение графа.")


    logger.info("Агент запущен")
    result = agent.invoke(
        {
            "repeat": settings.AGENT_REFINE_STEPS,
            "current_step": 0,
            "generation_prompt_client": generation_prompt_client,
            "refine_prompt_client": refine_prompt_client,
            "train_dataset_client": train_dataset,
            "test_dataset_client": test_dataset,
            "best_test_accuracy": best_test_accuracy,
        }
    )


def accuracy(input: str, output: dict[str, Any], expected_output: dict[str, Any]):
    scores = []
    bad_result: BadResult = {"generated_answers": {}, "reference_answers": {}}  

    for k, v in expected_output.items():
        if output[k] == v:
            scores.append(1)
        else:
            scores.append(0)
            bad_result["generated_answers"][k] = output[k]
            bad_result["reference_answers"][k] = expected_output[k]

    if bad_result:
        bad_result["text"] = input

    return sum(scores) / len(scores), bad_result


def generate_train(state: State):
    prompt_client = state["generation_prompt_client"]
    current_step = state["current_step"]
    train_dataset = state["train_dataset_client"]
    bad_results = []

    run_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    for item in tqdm(train_dataset.items[:10], desc="Generating training samples"):
        with langfuse_client.start_as_current_observation(
            name=f"generate-train-{current_step}", as_type="generation", prompt=prompt_client
        ):
            with item.run(run_name=f"run-{current_step}-{run_date}") as span:
                input_prompt = prompt_client.compile(document=item.input)
                output = generation_model.invoke(input=input_prompt, config={"callbacks": [callback_handler]})
                accuracy_score, bad_result = accuracy(item.input, output.model_dump(), item.expected_output)
                span.update_trace(input=input_prompt, output=output)
                span.score_trace(name="accuracy", value=accuracy_score)

        if bad_result:
            bad_results.append(bad_result)

    langfuse_client.flush()

    return {"bad_results": bad_results}


def generate_test(state: State):
    prompt_client = state["generation_prompt_client"]
    current_step = state["current_step"]
    # TODO: при первом запуске лучшая точность неизвестна, пока её добавляю вручную при вызове агента
    best_test_accuracy = state.get("best_test_accuracy", 0)
    test_dataset = state["test_dataset_client"]
    overall_accuracy = []

    run_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    for item in tqdm(test_dataset.items[:10], desc="Generating test samples"):
        with langfuse_client.start_as_current_observation(
            name=f"generate-test-{current_step}", as_type="generation", prompt=prompt_client
        ):
            with item.run(run_name=f"run-{current_step}-{run_date}") as span:
                input_prompt = prompt_client.compile(document=item.input)
                output = generation_model.invoke(input=input_prompt, config={"callbacks": [callback_handler]})
                accuracy_score, _ = accuracy(item.input, output.model_dump(), item.expected_output)
                overall_accuracy.append(accuracy_score)
                span.update_trace(input=input_prompt, output=output)
                span.score_trace(name="accuracy", value=accuracy_score)

    langfuse_client.flush()

    overall_accuracy = sum(overall_accuracy) / len(overall_accuracy)

    logger.info(f"{overall_accuracy=}, {best_test_accuracy=}")

    is_test_accuracy_improving = False
    if overall_accuracy > best_test_accuracy:
        best_test_accuracy = overall_accuracy
        is_test_accuracy_improving = True

    logger.info(f"{is_test_accuracy_improving=}, {prompt_client.version=}")

    return {"best_test_accuracy": best_test_accuracy, "is_test_accuracy_improving": is_test_accuracy_improving}


def refine(state: State):
    generation_prompt_client = state["generation_prompt_client"]
    refine_prompt_client = state["refine_prompt_client"]
    bad_results = state["bad_results"]
    current_step = state["current_step"]

    bad_examples = ""
    if len(bad_results) > 0:
        num_samples = min(len(bad_results), 5)
    for bad_result in random.sample(bad_results, k=num_samples):
        bad_examples += "## Документ\n"
        for k, generated in bad_result["generated_answers"].items():
            reference = bad_result["reference_answers"][k]
            bad_examples += (
                f"Поле для извлечения: {k}\nПравильный ответ: {reference}\nСгенерированный ответ: {generated}\n"
            )

    with langfuse_client.start_as_current_observation(
        name=f"refine-{current_step}", as_type="generation", prompt=refine_prompt_client
    ) as gen:
        input_prompt = refine_prompt_client.compile(
            generation_prompt=generation_prompt_client.compile(document="Пример документа"),
            json_schema=generation_prompt_client.config["json_schema"],
            examples=bad_examples,
        )
        output = refine_model.invoke(input=input_prompt, config={"callbacks": [callback_handler]})
        gen.update_trace(input=input_prompt, output=output["parsed"])

    langfuse_client.flush()

    new_generation_prompt_client = langfuse_client.create_prompt(
        name=generation_prompt_client.name,
        prompt=output["parsed"].new_prompt + "\n# Документ\n{{document}}",
        config=generation_prompt_client.config,
    )

    return {"messages": [output["raw"]], "generation_prompt_client": new_generation_prompt_client}


def update_generation_prompt(state: State):
    current_step = state["current_step"]
    is_test_accuracy_improving = state["is_test_accuracy_improving"]
    generation_prompt_client = state["generation_prompt_client"]

    current_step += 1
    if is_test_accuracy_improving:
        logger.info("Test accuracy is improving")
        logger.info(f"Promoting generation prompt v{generation_prompt_client.version} to 'best'")
        langfuse_client.update_prompt(
            name=generation_prompt_client.name,
            version=generation_prompt_client.version,
            new_labels=["best"],
        )
    else:
        generation_prompt_client = langfuse_client.get_prompt(
            name=generation_prompt_client.name,
            label="best",
        )
        logger.info("Test accuracy is not improving")
        logger.info(f"Loaded 'best' generation prompt v{generation_prompt_client.version}")
        
    return {
        "current_step": current_step,
        "generation_prompt_client": generation_prompt_client,
    }


def should_continue(state: State):
    current_step = state["current_step"]
    repeat = state["repeat"]

    if current_step >= repeat:
        return END
    return "continue"


def build_agent():

    builder = StateGraph(State)

    builder.add_node("generate_train", generate_train)
    builder.add_node("refine", refine)
    builder.add_node("generate_test", generate_test)
    builder.add_node("update_generation_prompt", update_generation_prompt)

    builder.add_edge(START, "generate_train")
    builder.add_edge("generate_train", "refine")
    builder.add_edge("refine", "generate_test")
    builder.add_edge("generate_test", "update_generation_prompt")
    builder.add_conditional_edges(
        "update_generation_prompt",
        should_continue,
        {
            END: END,
            "continue": "generate_train",
        },
    )

    agent = builder.compile()
    return agent


if __name__ == "__main__":
    main()