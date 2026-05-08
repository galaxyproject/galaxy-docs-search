#!/usr/bin/env python3

import argparse
from pathlib import Path


from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def search_chromadb_vectors(store_path: str, query: str, top_k: int = 5):

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        base_url="https://api.deepinfra.com/v1/openai",
        model="BAAI/bge-large-en-v1.5",
        api_key="",
        tiktoken_enabled=False,  # Disable tiktoken to avoid DNS issues with custom models
        check_embedding_ctx_length=False  # Disable context length checking
    )

    vectorstore = Chroma(
        persist_directory=store_path,
        collection_name="gtn_tutorials",
        embedding_function=embeddings
    )

    print("Using Chroma vector store for similarity search using query:", query)

    #results =  vectorstore.similarity_search(query, k=5)

    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    print(f"Found {len(results_with_scores)} similar documents")

    for i, (doc, score) in enumerate(results_with_scores, start=1):
        print("=" * 80)
        print(f"Document: {doc}")
        print(f"Metadata: {doc.metadata}")
        print(f"URL: {doc.metadata.get('url', 'Unknown')}")
        print(f"Content: {doc.page_content}")
        print(f"Score: {score: 0.3f}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_path", help="Path to persisted LlamaIndex vector store")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=1)

    args = parser.parse_args()

    search_chromadb_vectors(
        store_path=args.store_path,
        query=args.query,
        top_k=args.top_k,
    )