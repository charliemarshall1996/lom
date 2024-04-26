
import json


def read_json_mapping(file):
    with open(file, encoding="utf-8-sig", errors="ignore") as f:
        return json.load(f)


def txt_to_set(path: str) -> set[str]:
    with open(path, encoding='utf-8-sig', errors='ignore') as f:
        return {line for line in f.readlines()}
