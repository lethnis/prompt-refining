import random
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm

from constants import BadResult, State
from models import refine_model, structured_model
from src.langfuse_utils import get_generation_prompt_client, get_refine_prompt_client, get_train_test_dataset
from src.settings import Settings

load_dotenv()

langfuse_client = get_client()
callback_handler = CallbackHandler()


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
    train_dataset = state["train_dataset"]
    bad_results = []

    for item in tqdm(train_dataset.items[:10], desc="Generating training samples"):
        with langfuse_client.start_as_current_observation(
            name=f"generate-train-{current_step}", as_type="generation", prompt=prompt_client
        ):
            with item.run(run_name=f"run-{current_step}") as span:
                input_prompt = prompt_client.compile(document=item.input)
                output = structured_model.invoke(input=input_prompt, config={"callbacks": [callback_handler]})
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
    best_test_accuracy = state.get("best_test_accuracy", 0)
    test_dataset = state["test_dataset"]
    overall_accuracy = []

    for item in tqdm(test_dataset.items[:10], desc="Generating test samples"):
        with langfuse_client.start_as_current_observation(
            name=f"generate-test-{current_step}", as_type="generation", prompt=prompt_client
        ):
            with item.run(run_name=f"run-{current_step}") as span:
                input_prompt = prompt_client.compile(document=item.input)
                output = structured_model.invoke(input=input_prompt, config={"callbacks": [callback_handler]})
                accuracy_score, _ = accuracy(item.input, output.model_dump(), item.expected_output)
                overall_accuracy.append(accuracy_score)
                span.update_trace(input=input_prompt, output=output)
                span.score_trace(name="accuracy", value=accuracy_score)

    langfuse_client.flush()

    overall_accuracy = sum(overall_accuracy) / len(overall_accuracy)

    is_test_accuracy_improving = False
    if overall_accuracy > best_test_accuracy:
        best_test_accuracy = overall_accuracy
        is_test_accuracy_improving = True

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
    best_generation_prompt_version = state.get("best_generation_prompt_version", 1)

    current_step += 1
    if is_test_accuracy_improving:
        best_generation_prompt_version = generation_prompt_client.version
    else:
        generation_prompt_client = langfuse_client.get_prompt(
            name=generation_prompt_client.name,
            version=best_generation_prompt_version,
        )

    return {
        "current_step": current_step,
        "best_generation_prompt_version": best_generation_prompt_version,
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


def main():
    settings = Settings()

    generation_prompt_client = get_generation_prompt_client(langfuse_client, settings)
    refine_prompt_client = get_refine_prompt_client(langfuse_client, settings)

    train_dataset, test_dataset = get_train_test_dataset(langfuse_client, settings)

    agent = build_agent()

    if settings.DRAW_AGENT_GRAPH is not None:
        try:
            agent.get_graph().draw_mermaid_png(output_file_path=settings.DRAW_AGENT_GRAPH, max_retries=3)
        except:
            print("Не удалось сохранить изображение графа.")

    result = agent.invoke(
        {
            "repeat": 5,
            "current_step": 0,
            "generation_prompt_client": generation_prompt_client,
            "refine_prompt_client": refine_prompt_client,
            "train_dataset_client": train_dataset,
            "test_dataset_client": test_dataset,
        }
    )
