
from ast import literal_eval
import csv
import json

import pandas as pd


def read_json_mapping(file):
    with open(file, encoding="utf-8-sig", errors="ignore") as f:
        return json.load(f)


def txt_to_set(path: str) -> set[str]:
    with open(path, encoding='utf-8-sig', errors='ignore') as f:
        return {line for line in f.readlines()}


def csv_to_json(file, output_file):
    dct = {}
    with open(file, encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            k = row[0]
            v = row[1]
            dct[k] = v

        with open(output_file, encoding="utf-8-sig", mode="w", errors="ignore") as out:
            json.dump(dct, out, indent=4)


def evaluate_lyrics(x):

    try:
        return literal_eval(x)
    except (SyntaxError, ValueError):
        return None


def load_dataset():
    return pd.read_csv("./data/processed/song_lyrics_processed.csv", converters={'lyrics': evaluate_lyrics}, encoding="utf-8-sig", encoding_errors="ignore", low_memory=False, on_bad_lines='skip')


def load_genre_dataset(genre):
    df = pd.DataFrame()
    for i, c in enumerate(pd.read_csv("./data/processed/song_lyrics_processed.csv", encoding="utf-8-sig", encoding_errors="ignore", on_bad_lines='skip', chunksize=100000)):
        print(f"Loading {genre} chunk {i}...")
        c = c[c.loc[:, "tag"] == genre]
        c.loc[:, "lyrics"] = c['lyrics'].apply(evaluate_lyrics)
        c.dropna(how="any", inplace=True)
        df = pd.concat([df, c])
        print(f"{genre} chunk {i} loaded. N={len(df)}...")
    return df.lyrics


if __name__ == "__main__":
    in_file = "C:\\Users\\charl.DESKTOP-1NGH5IT\\Documents\\GitHub\\lom\\data\\vocab\\dropped_gs.csv"
    out_file = "C:\\Users\\charl.DESKTOP-1NGH5IT\\Documents\\GitHub\\lom\\data\\vocab\\dropped_gs.json"
    csv_to_json(in_file, out_file)
