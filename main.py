import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from character_parser import parse_png

nltk.download("punkt")
nltk.download("stopwords")


def process_folder(folder_path, output_file):
    characters = []

    for filename in os.listdir(f"characters/{folder_path}"):
        if filename.endswith(".png"):
            file_path = os.path.join(f"characters/{folder_path}", filename)

            with open(file_path, "rb") as f:
                png_data = f.read()

            character = parse_png(png_data)
            if character is not None:
                characters.append(character)

    with open(f"data/{output_file}", "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=4, ensure_ascii=False)


def load_and_preprocess_data(file_path):
    with open(f"data/{file_path}", "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        if "name" in item["data"] and "description" in item["data"]:
            char = f"{item['data']['name']} {item['data']['description']}"

            if item["data"]["first_mes"] != "":
                char += f" {item['data']['first_mes']}"

            if item["data"]["tags"] != []:
                char += f" {', '.join(item['data']['tags'])}"

            processed_data.append(char)

    return processed_data


def preprocess_text(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    word_tokens = word_tokenize(text.lower())
    filtered_text = [
        stemmer.stem(w.lower())
        for w in word_tokens
        if w.isalnum() and w not in stop_words
    ]

    return " ".join(filtered_text)


if (not os.path.exists("data/underage_characters.json")) or (
    not os.path.exists("data/adult_characters.json")
):
    process_folder("adult", "adult_characters.json")
    process_folder("underage", "underage_characters.json")


underage_data = load_and_preprocess_data("underage_characters.json")
adult_data = load_and_preprocess_data("adult_characters.json")

underage_labels = [1] * len(underage_data)
adult_labels = [0] * len(adult_data)

print(f"Adult: {len(adult_data)}")
print(f"Underage: {len(underage_data)}")

labels = underage_labels + adult_labels
all_data = underage_data + adult_data

preprocessed_data = [preprocess_text(text) for text in all_data]

X_train, X_test, y_train, y_Test = train_test_split(
    preprocessed_data, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = SVC(kernel="linear")
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
print(classification_report(y_Test, y_pred))


def is_underage(text: str) -> bool:
    preprocessed = preprocess_text(text)
    vectorized = vectorizer.transform([preprocessed])
    prediction = model.predict(vectorized)

    return True if prediction[0] == 1 else False


def classify_character(character: dict) -> bool:
    char = f"{character['data']['name']} {character['data']['description']}"

    if character["data"]["first_mes"] != "":
        char += f" {character['data']['first_mes']}"

    if character["data"]["tags"] != []:
        char += f" {', '.join(character['data']['tags'])}"

    return is_underage(char)


def test_character_classification(char_type: str):
    characters = []

    for filename in os.listdir(f"test/{char_type}"):
        if filename.endswith(".png"):
            file_path = os.path.join(f"test/{char_type}", filename)

            with open(file_path, "rb") as f:
                png_data = f.read()

            character = parse_png(png_data)
            if character is not None:
                characters.append(character)

    for character in characters:
        print(f"{character['data']['name']} - is underage? {classify_character(character)}")


test_character_classification("adult")
test_character_classification("underage")