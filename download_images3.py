import os
import json
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0 Safari/537.36",
    "Referer": "https://www.furtrack.com/"
}

session = requests.Session()
session.headers.update(HEADERS)

def download_image(post_id, character_name, save_dir):
    img_url = f"https://orca2.furtrack.com/thumb/{post_id}.jpg"
    save_path = os.path.join(save_dir, f"{character_name}_{post_id}.jpg")

    try:
        response = session.get(img_url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[{character_name}] Downloaded image {post_id}")
    except Exception as e:
        print(f"Failed to download {post_id}: {e}")

def download_from_local_json(character_name, base_folder="dataset"):
    char_folder = os.path.join(base_folder, character_name)
    json_path = os.path.join(char_folder, f"{character_name}.json")

    if not os.path.isfile(json_path):
        print(f"JSON file not found for {character_name} at {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON for {character_name}: {e}")
        return

    posts = data.get("posts", [])
    print(f"[{character_name}] Found {len(posts)} posts.")

    for post in posts:
        post_id = post.get("postId")
        if post_id:
            download_image(post_id, character_name, char_folder)

def batch_download(characters):
    for character in characters:
        download_from_local_json(character)

if __name__ == "__main__":
    characters = [
        "tempo_(arcanine)"
    ]
    batch_download(characters)
