#!/usr/bin/env python
# This is a modified version of https://github.com/hpsee/discourse-cluster/blob/master/export_posts.py
# It has been enhanced for robustness, local caching, and incremental updates.

from datetime import datetime
import json
import os
from time import sleep
import sys
import re
import requests
from hashlib import sha256  # For more robust hashing


# --- Configuration ---
export_base_folder = os.path.abspath("posts")
cache_folder = os.path.join(export_base_folder, ".cache")  # Hidden cache folder
sleep_time = 1.5
api_key = os.environ.get("DISCOURSE_API_KEY")
username = os.environ.get("DISCOURSE_API_USER")
api_base = os.environ.get("DISCOURSE_BASE", "https://help.galaxyproject.org")
category = os.environ.get("DISCOURSE_CATEGORY", "usegalaxy-org-support")

# --- Input Validation and Setup ---

if not api_key or not username:
    sys.exit("Please export your DISCOURSE_API_KEY and DISCOURSE_API_USER")

if not api_base.startswith(("http://", "https://")):
    sys.exit("DISCOURSE_BASE must start with http:// or https://")

if not os.path.exists(export_base_folder):
    print(f"Creating export directory: {export_base_folder}")
    os.makedirs(export_base_folder)  # Create recursively

if not os.path.exists(cache_folder):
    print(f"Creating cache directory: {cache_folder}")
    os.makedirs(cache_folder)


print(f"Category is {category}")
print(f"API Base URL: {api_base}")


# --- Helper Functions ---


def write_json(json_object, output_file, mode="w"):
    """Pretty print a json object to file."""
    with open(output_file, mode) as filey:
        json.dump(json_object, filey, indent=4)


def read_json(input_file):
    """Read a json file, handling potential errors."""
    try:
        with open(input_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_response_hash(response):
    """Generate a SHA256 hash of the response content."""
    return sha256(response.content).hexdigest()


def fetch_with_cache(url, headers, cache_key_prefix, use_cache=True):
    """Fetches data from a URL, using and updating a local cache."""
    cache_key = f"{cache_key_prefix}-{sha256(url.encode('utf-8')).hexdigest()}.json"
    cache_path = os.path.join(cache_folder, cache_key)

    if use_cache:
        cached_data = read_json(cache_path)
        if cached_data:
            print(f"Using cached data for {url}")
            return cached_data, True  # Return data and a flag indicating cache hit

    print(f"Fetching {url}")
    try:
        response = requests.get(url, headers=headers, timeout=30)  # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        write_json(data, cache_path)  # Cache the successful response
        return data, False  # Return data and cache miss flag
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, False


# --- Main Logic ---


def fetch_topics_and_tags():
    """Fetches the list of topics and top tags for the category."""
    url = f"{api_base}/c/{category}.json"
    headers = {"Api-Key": api_key, "Api-Username": username}
    topics = []
    tags = []

    while url:
        data, cache_hit = fetch_with_cache(url, headers, "topic_list")
        if not data:
            return [], []  # Return empty lists on failure

        results = data.get("topic_list", {})
        topics.extend(results.get("topics", []))
        tags.extend(results.get("top_tags", []))

        url = results.get("more_topics_url")
        if url:
            url = f"{api_base}{url}"  # Construct full URL
            if not url.endswith(".json"):  # Ensure .json suffix for consistency
                url_part, param_part = url.split("?", 1)
                url = f"{url_part}.json?{param_part}"
            sleep(sleep_time)  # Respect API rate limits

    return topics, tags


def fetch_and_save_post(topic, export_folder, headers):
    """Fetches and saves an individual post, handling caching."""
    url = f"{api_base}/t/{topic['slug']}/{topic['id']}.json"
    post_id = f"post-{topic['id']}"
    export_path = os.path.join(export_folder, f"{post_id}.json")

    post_data, cache_hit = fetch_with_cache(url, headers, post_id)
    if post_data:
        write_json(post_data, export_path)
    return post_data


def main():
    """Main execution function."""
    topics, tags = fetch_topics_and_tags()

    if not topics:
        print("No topics found. Exiting.")
        return

    print(f"Found {len(topics)} topics, and {len(tags)} top tags!")

    # Create a folder within data organized by date
    now = datetime.now()
    date_folder = f"{now.year}-{now.month}-{now.day}"
    export_folder = os.path.join(export_base_folder, date_folder)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    prefix = re.sub(r"(http://|https://)", "", api_base) + f"-{category}-{date_folder}-"
    export_path = os.path.join(export_folder, prefix)

    # Save topics and tags
    write_json(topics, f"{export_path}topics.json")
    write_json(tags, f"{export_path}tags.json")

    # Keep a lookup of all post text by id, loading existing data
    posts_file = f"{export_path}posts.json"
    posts = read_json(posts_file) or {}

    headers = {"Api-Key": api_key, "Api-Username": username}
    # Fetch and save individual posts
    for topic in topics:
        post_data = fetch_and_save_post(topic, export_folder, headers)

        if post_data:
            # Only update posts if the data has changed.  Use cached data, if available, to check.
            existing_content = posts.get(str(post_data["id"]))
            new_content = [x["cooked"] for x in post_data["post_stream"]["posts"]]

            if existing_content != new_content:  # Compare content
                posts[str(post_data["id"])] = (
                    new_content  # Use string keys for consistency
                )
                print(f"  Updating post content for topic ID {post_data['id']}")
        sleep(sleep_time)

    # Save updated post content
    write_json(posts, posts_file)
    print("Export complete.")


if __name__ == "__main__":
    main()
