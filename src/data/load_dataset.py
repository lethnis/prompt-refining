from datasets import load_dataset
from datetime import date

dataset = load_dataset("irlspbru/RusLawOD", data_files=["ruslawod_01.parquet"])
dataset = dataset.filter(lambda x: x["textIPS"] is not None).map(lambda x: {"word_count": len(x["textIPS"].split())})
dataset = dataset.remove_columns(["pravogovruNd", "actual_datetimeIPS", "actual_datetime_humanIPS", "taggedtextIPS", "keywordsByIPS", "classifierByIPS", "is_widely_used", "statusIPS"])
dataset = dataset.filter(lambda x: x["word_count"] > 50 and x["word_count"] < 500)
df = dataset["train"].to_polars()
keep_doc_types = set(df.group_by("doc_typeIPS").len().sort("len", descending=True)[:5]["doc_typeIPS"])
dataset = dataset.filter(lambda x: x["doc_typeIPS"] in keep_doc_types)
dataset = dataset.class_encode_column("doc_typeIPS")
dataset = dataset["train"].train_test_split(test_size=100, train_size=200, stratify_by_column="doc_typeIPS")
dataset = dataset.map(lambda x: {"doc_type": dataset["train"].features["doc_typeIPS"].int2str(x["doc_typeIPS"])}).remove_columns(["doc_typeIPS"])
dataset = dataset.rename_columns({
    "issuedByIPS": "full_name",
    "docdateIPS": "publication_date",
    "docNumberIPS": "number",
    "headingIPS": "title",
    "doc_author_normal_formIPS": "government_agency_name",
    "signedIPS": "signatory",
    "doc_type": "type",
    "textIPS": "text",
    }).remove_columns(["word_count"])
def parse_date(x):
    date_str = x["publication_date"]
    day, month, year = date_str.split(".")
    return {"publication_date": date(int(year), int(month), int(day))}
dataset = dataset.map(parse_date)
dataset.save_to_disk("../../data/raw/")
