import json
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np
import onnx
from onnx import version_converter, helper
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization import quantize_dynamic, QuantType

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

labels = underage_labels + adult_labels
all_data = underage_data + adult_data

preprocessed_data = [preprocess_text(text) for text in all_data]

X_train, X_test, y_train, y_Test = train_test_split(
    preprocessed_data, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = SVC(kernel="linear", probability=True)
model.fit(X_train_vectorized, y_train)

y_pred_proba = model.predict_proba(X_test_vectorized)[:, 1]
y_pred = model.predict(X_test_vectorized)
print(classification_report(y_Test, y_pred))

roc_auc = roc_auc_score(y_Test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")


def is_underage(text: str):
    preprocessed = preprocess_text(text)
    vectorized = vectorizer.transform([preprocessed])
    proba = model.predict_proba(vectorized)[0, 1]

    score = proba.item() if hasattr(proba, "item") else proba

    return score


def classify_character(character: dict):
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
        print(
            f"{character['data']['name']} - is underage? {classify_character(character) > 0.5}"
        )


test_character_classification("adult")
test_character_classification("underage")


def convert_to_onnx():
    initial_type = [
        ("float_input", FloatTensorType([None, X_train_vectorized.shape[1]]))
    ]
    onx = convert_sklearn(model, initial_types=initial_type)

    with open("underage_classifier.onnx", "wb") as f:
        f.write(onx.SerializeToString())


if not os.path.exists("underage_classifier.onnx"):
    convert_to_onnx()

sess = rt.InferenceSession("underage_classifier.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


def predict_onnx(text):
    preprocessed = preprocess_text(text)
    vectorized = vectorizer.transform([preprocessed]).toarray().astype(np.float32)
    pred_onx = sess.run([label_name], {input_name: vectorized})[0]
    return pred_onx[0]


def is_underage_onnx(text: str) -> bool:
    preprocessed = preprocess_text(text)
    vectorized = vectorizer.transform([preprocessed]).toarray().astype(np.float32)
    prediction = sess.run([label_name], {input_name: vectorized})[0]

    return True if prediction[0] == 1 else False


def classify_character_onnx(character: dict) -> bool:
    char = f"{character['data']['name']} {character['data']['description']}"

    if character["data"]["first_mes"] != "":
        char += f" {character['data']['first_mes']}"

    if character["data"]["tags"] != []:
        char += f" {', '.join(character['data']['tags'])}"

    return is_underage_onnx(char)


def test_character_classification_onnx(char_type: str):
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
        print(
            f"{character['data']['name']} - is underage? {classify_character_onnx(character)}"
        )
