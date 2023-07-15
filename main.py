import os
import re
import frontmatter
import openai
import pinecone
import tiktoken


# Find all GTN markdown Files
def find_md_files(directory: str) -> list:
    md_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("tutorial.md"):
                file_path = os.path.join(root, file)
                md_files.append(file_path)

    return md_files


# Open the markdown files in frontmatter for initial parsing
def remove_original_metadata(path: str) -> object:
    post = frontmatter.load(path)
    return post

# Separate Markdown files by section
def split_by_single_hash(string: str) -> list:
    sections = []
    current_section = []

    lines = string.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#") and line.count("#") == 1:
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
        current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    return sections


# Clean markdown files
def clean_regex(input: list) -> list:
    parsed = []
    for i in input:
        cleaned = re.sub(r"\{([^}]*)\}", "", i)
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub(r"\[.*?\]", "", cleaned)
        cleaned = re.sub(r"\(.*?\)", "", cleaned)
        cleaned = re.sub(r"\<.*?\>", "", cleaned)
        cleaned = cleaned.replace(">", "")
        cleaned = cleaned.replace("  ", " ")
        cleaned = cleaned.replace("#", "")
        parsed.append(cleaned)
    return parsed


# Construct the URL for metadata section of vector store
def url_constructor(
    path: str, target: str, original_file_extension: str, new_file_extension: str
) -> str:
    stable_path = "https://training.galaxyproject.org/training-material/topics"
    new_target = path.split(target, 1)[1]
    combined_path = stable_path + new_target
    combined_path = combined_path.replace(original_file_extension, new_file_extension)
    return combined_path


# Combine the introduction and final paragraph of markdown files
def combine_intro_conclusion(input: list) -> list:
    input_element = str(input[0]) + str(input[-1])
    input.pop(0)
    input.pop(-1)
    input.insert(0, input_element)
    return input


# Estimate the number of tokens that would be used to encode a string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Split a string in half recrurisvely until it is a certain size
def split_string_recursive(string: str, size: int, overlap: int) -> str:
    if num_tokens_from_string(string, "cl100k_base") <= size:
        return [string]
    mid = len(string) // 2
    left_half = split_string_recursive(string[: mid + overlap], size, overlap)
    right_half = split_string_recursive(string[mid - overlap :], size, overlap)

    return left_half + right_half


# Force al lchunks to be below a certain size
def standardize_chunk_size(input: list, chunk_size: int, overlap: int) -> list:
    standard_chunk_list = []
    for i in input:
        standard_chunk_list.extend(split_string_recursive(i, chunk_size, overlap))
    return standard_chunk_list


# Build full meta data object
def build_vector_metadata(
    path: str, target: str, orig_ext: str, new_ext: str, name: str
) -> dict:
    metadata_dict = {}
    metadata_dict["url"] = url_constructor(path, target, orig_ext, new_ext)
    metadata_dict["name"] = name
    return metadata_dict


# Create Embedding from text
def embed(text: list, model: str) -> object:
    res = openai.Embedding.create(input=text, engine=model)

    embeds = [record["embedding"] for record in res["data"]]
    return embeds


# Create final zip object to upload to pinecone
def final_vector_zip(embed: list, meta: dict) -> list:
    meta_list = []
    id_list = []
    for i in range(len(embed)):
        meta_list.append(meta)
    for i in range(len(embed)):
        url = meta["url"]
        url = str(url) + str(i)
        id_list.append(url)
    return zip(id_list, embed, meta_list)


def main():
    directory = "/home/tcollins/training-material/topics"
    openai.api_key = ""
    MODEL = "text-embedding-ada-002"
    index_name = "gtn-tutorials"
    PINECONE_API_KEY = ""
    PINECONE_ENV = "asia-southeast1-gcp-free"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeds[0]))
    index = pinecone.Index(index_name)

    markdown_files = find_md_files(directory)
    for file in markdown_files:
        post = remove_original_metadata(file)
        tutorial_by_sections = split_by_single_hash(post.content)
        cleaned_tutorial_by_sections = clean_regex(tutorial_by_sections)
        # combined_clean_tutorials = combine_intro_conclusion(cleaned_tutorial_by_sections)
        combined_clean_sized_tutorials = standardize_chunk_size(
            cleaned_tutorial_by_sections, 1000, 200
        )
        embeddings = embed(combined_clean_sized_tutorials, MODEL)
        meta = build_vector_metadata(
            file, "topics", ".md", ".html", post.metadata["title"]
        )
        to_upsert = final_vector_zip(embeddings, meta)
        index.upsert(vectors=list(to_upsert))


main()
