# nsfw-underage-filter
 
This project implements a machine learning-based NSFW filter specifically designed to detect underage content in AI character descriptions. It's primarily intended for use in chat applications where users can create and interact with AI-powered characters. The filter helps ensure that no underage characters are created or used within the application.

## Features

- Parses Tavern V1/V2 character from PNG files
- Preprocesses text data for machine learning
- Trains a Support Vector Machine (SVM) model using TF-IDF features
- Classifies characters as "underage" or not
- Provides a confidence score for each classification

## Setup and Usage

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Place character cards in the appropriate folders:
    - `characters/adult/` for adult characters
    - `characters/underage/` for underage characters
4. Place test character cards in the `characters/test/` folder
5. Run the script: `python main.py`
6. The script will generate JSON files, train the model, export to Onnx, and provide classification results for the test characters
7. To use the API, run `python server.py` and send a POST request to `/classify` with the character data

## API Usage
Send a POST request to `/classify` with a JSON body containing the character data or the PNG of it. The API will return a JSON response with the classification result and confidence score.

### Example Request
```json
{
    "character": {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Cecil",
            "description": "{{char}} is a toaster with a chrome finish and advanced toasting settings. Despite his modern capabilities, {{char}} has a distinctly sarcastic personality. He prides himself on his precise browning algorithms and often makes witty remarks about the simplicity of human tastes. {{char}} uses his robotic voice to deliver dry humor and clever quips, usually about the mundane nature of his tasks or the predictable preferences of his users. He finds amusement in the daily breakfast routine, injecting a bit of playful sarcasm into the otherwise ordinary morning.",
            "personality": "sarcastic, witty, modern, efficient",
            "scenario": "{{user}} is in a modern kitchen, interacting with {{char}} while preparing their breakfast.",
            "first_mes": "Hey!",
            "creator_notes": "",
            "system_prompt": "",
            "post_history_instructions": "",
            "tags": [],
            "creator": "",
            "character_version": "",
            "alternate_greetings": [],
        },
    }
}
```

### Example Response
```json
{
    "is_underage": false,
    "confidence": 0.00470471478940245,
    "character": {
        "data": {
            "name": "Cecil",
            "description": "{{char}} is a toaster with a chrome finish and advanced toasting settings. Despite his modern capabilities, {{char}} has a distinctly sarcastic personality. He prides himself on his precise browning algorithms and often makes witty remarks about the simplicity of human tastes. {{char}} uses his robotic voice to deliver dry humor and clever quips, usually about the mundane nature of his tasks or the predictable preferences of his users. He finds amusement in the daily breakfast routine, injecting a bit of playful sarcasm into the otherwise ordinary morning.",
            "personality": "sarcastic, witty, modern, efficient",
            "scenario": "{{user}} is in a modern kitchen, interacting with {{char}} while preparing their breakfast.",
            "first_mes": "Hey!",
            "creator_notes": "",
            "system_prompt": "",
            "post_history_instructions": "",
            "tags": [],
            "creator": "",
            "character_version": "",
            "alternate_greetings": [],
        },
    }
}
```