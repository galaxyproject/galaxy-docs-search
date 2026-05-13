#!/usr/bin/env python3

import argparse
import os
from pathlib import Path


from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


def search_chromadb_vectors(store_path: str, query: str, top_k: int = 5):

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        base_url=os.getenv("EMBEDDINGS_BASE_URL"),
        model=os.getenv("EMBEDDINGS_MODEL"),
        api_key=os.getenv("EMBEDDINGS_API_KEY"),
        tiktoken_enabled=False,
        check_embedding_ctx_length=False
    )

    gtn_vectorstore = Chroma(
        persist_directory=store_path,
        collection_name="gtn_tutorials",
        embedding_function=embeddings
    )

    iwc_vectorstore = Chroma(
        persist_directory=store_path,
        collection_name="iwc_workflows",
        embedding_function=embeddings
    )

    galaxy_tools_vectorstore = Chroma(
        persist_directory=store_path,
        collection_name="galaxy_tools",
        embedding_function=embeddings
    )

    print("Using Chroma vector store for similarity search using query:", query)

    results_with_scores = {
        'iwc': iwc_vectorstore.similarity_search_with_score(query, k=top_k),
        'gtn': gtn_vectorstore.similarity_search_with_score(query, k=top_k),
        'galaxy_tools': galaxy_tools_vectorstore.similarity_search_with_score(query, k=top_k),
    }

    print(f"Found {len(results_with_scores['iwc'])} similar IWC documents and {len(results_with_scores['gtn'])} similar GTN documents")

    print("\nTop similar IWC documents with scores:")

    for i, (doc, score) in enumerate(results_with_scores['iwc'], start=1):
        print("=" * 80)
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print(f"Score: {score: 0.3f}")
        print("=" * 80)

    print("\nTop similar GTN documents with scores:")

    for i, (doc, score) in enumerate(results_with_scores['gtn'], start=1):
        print("=" * 80)
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print(f"Score: {score: 0.3f}")
        print("=" * 80)

    print("\nTop similar Galaxy Tools documents with scores:")
    for i, (doc, score) in enumerate(results_with_scores['galaxy_tools'], start=1):
        print("=" * 80)
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print(f"Score: {score: 0.3f}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_path", help="Path to persisted LlamaIndex vector store")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    search_chromadb_vectors(
        store_path=args.store_path,
        query=args.query,
        top_k=args.top_k,
    )