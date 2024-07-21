import json
import os
import random
import string
from flask import Flask, request, jsonify

from character_parser import parse_png
from main import classify_character

app = Flask(__name__)


def generate_random_filename(length=8):
    letters_and_digits = string.ascii_letters + string.digits
    filename = "".join(random.choice(letters_and_digits) for i in range(length))
    return filename


@app.route("/classify", methods=["POST"])
def classify():
    if "character" in request.files:
        file = request.files["character"]
        filename = generate_random_filename(length=24) + ".png"
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        file_open = open(file_path, "rb")
        character = parse_png(file_open.read())
        file_open.close()
        
        if os.path.exists(file_path):
            os.remove(file_path)

        if character is not None:
            is_underage = classify_character(character)

            return jsonify(
                {
                    "is_underage": is_underage > 0.5,
                    "confidence": is_underage,
                    "character": character,
                }
            )


    data = request.get_json()

    if "character" not in data:
        return jsonify({"error": "Missing 'character' in request body"}), 400

    character = data["character"]
    is_underage = classify_character(character)

    return jsonify(
        {
            "is_underage": is_underage > 0.5,
            "confidence": is_underage,
            "character": character,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
