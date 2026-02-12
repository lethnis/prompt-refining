from datasets import load_from_disk
from langfuse import get_client
from tqdm import tqdm

from src.constants import ActInfo

client = get_client()
dataset = load_from_disk("../data/raw")
client.create_dataset(name="RusLawOD/train", expected_output_schema=ActInfo.model_json_schema())
client.create_dataset(name="RusLawOD/test", expected_output_schema=ActInfo.model_json_schema())

for item in tqdm(dataset["train"], desc="Loading train dataset..."):

    client.create_dataset_item(
        dataset_name="RusLawOD/train",
        input=item["text"],
        expected_output={
            "full_name": item["full_name"],
            "publication_date": item["publication_date"],
            "number": item["number"],
            "title": item["title"],
            "government_agency_name": item["government_agency_name"],
            "signatory": item["signatory"],
            "type": item["type"],
        },
    )

for item in tqdm(dataset["test"], desc="Loading test dataset..."):

    client.create_dataset_item(
        dataset_name="RusLawOD/test",
        input=item["text"],
        expected_output={
            "full_name": item["full_name"],
            "publication_date": item["publication_date"],
            "number": item["number"],
            "title": item["title"],
            "government_agency_name": item["government_agency_name"],
            "signatory": item["signatory"],
            "type": item["type"],
        },
    )
