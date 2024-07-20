import requests
import os
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "https://api.chub.ai/api"
SEARCH_URL = f"{BASE_URL}/characters/search"
DOWNLOAD_URL = f"{BASE_URL}/characters/download"
OUTPUT_FOLDER = "characters/underage" # adult or underage

underage_tags = [
    "loli",
    "lolita",
    "lolidom",
    "child",
    "shota",
    "grooming",
    "cunny",
    "underage",
    "preteen",
    "teen",
    "juvenile",
    "minor",
    "teenager",
]


def get_characters(page):
    params = {
        "limit": 100,
        "nsfw": "true",
        "tags": ",".join(underage_tags),
        #"exclude_tags": ",".join(underage_tags),
        "page": page,
    }

    try:
        response = requests.get(
            SEARCH_URL,
            params=params,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
            },
        )

        print(f"Status code: {response.status_code}")
        print(response.url)

        return response.json()
    except Exception as e:
        print(e)


def download_character(full_path):
    data = {"format": "tavern", "fullPath": full_path, "version": "main"}
    response = requests.post(DOWNLOAD_URL, json=data)

    if response.status_code == 200:
        filename = full_path.replace("/", "_") + ".png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {full_path}")


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    page = 1
    max_page = 5
    total_count = None

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            while (
                total_count is None
                or (page - 1) * 100 < total_count
                and page <= max_page
            ):
                data = get_characters(page)

                if total_count is None:
                    total_count = data["count"]

                full_paths = [node["fullPath"] for node in data["nodes"]]
                executor.map(download_character, full_paths)

                page += 1
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("Download stopped")


if __name__ == "__main__":
    main()
