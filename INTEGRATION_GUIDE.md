# ChatGXY Integration Guide

Complete guide for integrating the ChatGXY Materials Pipeline into your RAG system.

## System Overview

The ChatGXY Materials Pipeline provides four separate vector store collections containing Galaxy Project documentation:

| Collection | Content | Documents | Chunks | Use Case |
|------------|---------|-----------|--------|----------|
| `galaxy_faqs` | FAQ Q&A pairs | 25 | 62 | Quick answers to common questions |
| `gtn_tutorials` | Tutorial documentation | 168 | 9,295 | Detailed how-to guides |
| `discourse_posts` | Forum discussions | 5,986 | 70,516 | Community solutions & troubleshooting |
| `iwc_workflows` | Curated workflows | 105 | 772 | Analysis workflow documentation |

**Total**: 6,284 documents → 81,645 embedded chunks

## Quick Integration

### Basic Setup

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LMStudioEmbeddings

# Initialize embeddings (must match the embedding model used to create the stores)
embeddings = LMStudioEmbeddings(
    base_url="http://localhost:1234/v1",
    model="text-embedding-nomic-embed-text-v2-moe"
)

# Load collections
faq_store = Chroma(
    collection_name="galaxy_faqs",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

gtn_store = Chroma(
    collection_name="gtn_tutorials",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

discourse_store = Chroma(
    collection_name="discourse_posts",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

iwc_store = Chroma(
    collection_name="iwc_workflows",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)
```

## Collection Schemas

### 1. FAQ Collection (`galaxy_faqs`)

**Document Format**:
```
Q: {question}

A: {answer}
```

**Metadata Schema**:
```python
{
    "source": str,              # Path to faq.md
    "data_source": "gtn_faq",   # Always "gtn_faq"
    "content_type": "faq",      # Always "faq"
    "category": str,            # e.g. "For Individual Learners"
    "question": str,            # The actual question text
    "faq_index": int            # Position in FAQ file
}
```

**Best For**: Quick, direct answers to common questions

**Example Query**:
```python
results = faq_store.similarity_search("How do I use workflows?", k=3)
for result in results:
    print(f"Q: {result.metadata['question']}")
    print(f"Category: {result.metadata['category']}")
    print(f"A: {result.page_content}")
```

### 2. GTN Tutorials Collection (`gtn_tutorials`)

**Document Format**: Full tutorial markdown content

**Metadata Schema**:
```python
{
    "source": str,              # Path to tutorial.md file
    "data_source": "gtn",       # Always "gtn"
    "content_type": "tutorial", # Always "tutorial"
    "title": str,               # Tutorial title
    "topic": str,               # e.g. "assembly", "sequence-analysis"
    "subtopic": str,            # Subtopic if available
    "tags": str,                # Comma-separated tags
    "contributors": str         # Comma-separated author names
}
```

**Best For**: In-depth educational content, step-by-step guides

**Example Query**:
```python
results = gtn_store.similarity_search("RNA-seq analysis", k=5)
for result in results:
    print(f"Title: {result.metadata['title']}")
    print(f"Topic: {result.metadata['topic']}")
    print(f"Tags: {result.metadata['tags']}")
```

### 3. Discourse Posts Collection (`discourse_posts`)

**Document Format**: Topic title + original post + all replies

**Metadata Schema**:
```python
{
    "source": str,                    # discourse://category/topic_id
    "data_source": "discourse",       # Always "discourse"
    "content_type": "forum_post",     # Always "forum_post"
    "topic_id": int,                  # Discourse topic ID
    "title": str,                     # Topic title
    "slug": str,                      # URL slug
    "category": str,                  # e.g. "usegalaxy-org-support"
    "url": str,                       # Full URL to topic
    "reply_count": int,               # Number of replies
    "views": int,                     # View count
    "created_at": str,                # ISO timestamp
    "tags": str                       # Comma-separated tags
}
```

**Best For**: Real-world problems, community solutions, troubleshooting

**Example Query**:
```python
results = discourse_store.similarity_search("workflow error", k=5)
for result in results:
    print(f"Title: {result.metadata['title']}")
    print(f"Category: {result.metadata['category']}")
    print(f"Replies: {result.metadata['reply_count']}")
    print(f"URL: {result.metadata['url']}")
```

### 4. IWC Workflows Collection (`iwc_workflows`)

**Document Format**: Workflow name + README + annotations + step documentation

**Metadata Schema**:
```python
{
    "source": str,                # iwc://path/to/workflow
    "data_source": "iwc",         # Always "iwc"
    "content_type": "workflow",   # Always "workflow"
    "workflow_name": str,         # Human-readable workflow name
    "workflow_id": str,           # IWC workflow ID
    "topic": str,                 # e.g. "computational-chemistry"
    "path": str,                  # Directory path
    "categories": str,            # Comma-separated (COVID-19, Structural Biology, etc.)
    "collections": str,           # Comma-separated (Computational Chemistry, etc.)
    "authors": str,               # Comma-separated author names
    "orcids": str,                # Comma-separated ORCID identifiers
    "license": str,               # License (e.g. "MIT")
    "release": str,               # Version number
    "doi": str,                   # DOI for citation
    "updated": str,               # Last updated date
    "url": str                    # Full URL to IWC page
}
```

**Best For**: Finding analysis workflows, discovering methods, workflow documentation

**Example Query**:
```python
results = iwc_store.similarity_search("RNA-seq analysis", k=5)
for result in results:
    print(f"Workflow: {result.metadata['workflow_name']}")
    print(f"Topic: {result.metadata['topic']}")
    print(f"Categories: {result.metadata['categories']}")
    print(f"URL: {result.metadata['url']}")
    print(f"DOI: {result.metadata['doi']}")
```

**Example Queries with Great Results**:
- "RNA-seq analysis" → RNA-Seq workflows (single-end, paired-end, differential expression)
- "variant calling" → Viral and COVID-19 variant calling workflows
- "metagenomics" → MAGs generation, metaproteomics
- "ChIP-seq" → ChIP-seq analysis workflows (single/paired-end, consensus peaks)
- "molecular dynamics" → GROMACS workflows, protein-ligand parameterization
- "single cell sequencing" → 10X Genomics, Scanpy workflows
- "drug design" → Fragment-based docking, virtual screening

## Retrieval Strategies

### Strategy 1: Prioritized Search (Recommended)

Search collections in priority order: FAQs → Workflows → Tutorials → Discourse

```python
def prioritized_search(query: str, k: int = 5):
    """Search with priority: FAQ > IWC > GTN > Discourse"""
    results = {
        'faq': faq_store.similarity_search(query, k=k),
        'workflows': iwc_store.similarity_search(query, k=k),
        'tutorials': gtn_store.similarity_search(query, k=k),
        'community': discourse_store.similarity_search(query, k=k)
    }
    return results

# Usage
results = prioritized_search("How do I install Galaxy?")

# Present in priority order
if results['faq']:
    print("Quick Answer:")
    print(results['faq'][0].page_content)

print("\nRelevant Workflows:")
for doc in results['workflows'][:3]:
    print(f"- {doc.metadata['workflow_name']}")

print("\nDetailed Guides:")
for doc in results['tutorials'][:3]:
    print(f"- {doc.metadata['title']}")

print("\nCommunity Discussions:")
for doc in results['community'][:3]:
    print(f"- {doc.metadata['title']}")
```

### Strategy 2: Merged Results with Scores

Combine all sources and re-rank by relevance

```python
def merged_search(query: str, k: int = 15):
    """Search all collections and merge results"""
    faq_results = faq_store.similarity_search_with_score(query, k=k)
    iwc_results = iwc_store.similarity_search_with_score(query, k=k)
    gtn_results = gtn_store.similarity_search_with_score(query, k=k)
    discourse_results = discourse_store.similarity_search_with_score(query, k=k)

    # Combine with source tracking
    all_results = []
    for doc, score in faq_results:
        all_results.append((doc, score, 'faq'))
    for doc, score in iwc_results:
        all_results.append((doc, score, 'workflow'))
    for doc, score in gtn_results:
        all_results.append((doc, score, 'tutorial'))
    for doc, score in discourse_results:
        all_results.append((doc, score, 'community'))

    # Sort by score (lower is better for cosine distance)
    all_results.sort(key=lambda x: x[1])

    return all_results[:k]

# Usage
results = merged_search("workflow error", k=10)
for doc, score, source in results:
    title = doc.metadata.get('title', doc.metadata.get('question', 'Untitled'))
    print(f"[{source}] {title} (score: {score:.3f})")
```

### Strategy 3: Weighted Search

Apply different weights to different sources

```python
def weighted_search(query: str, k: int = 10, weights={'faq': 2.0, 'iwc': 1.8, 'gtn': 1.5, 'discourse': 1.0}):
    """Search with source-based weighting"""
    faq_results = faq_store.similarity_search_with_score(query, k=k*2)
    iwc_results = iwc_store.similarity_search_with_score(query, k=k*2)
    gtn_results = gtn_store.similarity_search_with_score(query, k=k*2)
    discourse_results = discourse_store.similarity_search_with_score(query, k=k*2)

    # Apply weights (boost FAQ, workflows, and tutorials)
    all_results = []
    for doc, score in faq_results:
        weighted_score = score / weights['faq']  # Lower score = better
        all_results.append((doc, weighted_score, 'faq'))
    for doc, score in iwc_results:
        weighted_score = score / weights['iwc']
        all_results.append((doc, weighted_score, 'workflow'))
    for doc, score in gtn_results:
        weighted_score = score / weights['gtn']
        all_results.append((doc, weighted_score, 'tutorial'))
    for doc, score in discourse_results:
        weighted_score = score / weights['discourse']
        all_results.append((doc, weighted_score, 'community'))

    # Sort and return top k
    all_results.sort(key=lambda x: x[1])
    return all_results[:k]
```

### Strategy 4: Filtered Search

Search specific sources based on query type

```python
def smart_search(query: str, k: int = 5):
    """Choose source based on query type"""
    query_lower = query.lower()

    # Question words → prioritize FAQ
    if any(word in query_lower for word in ['how', 'what', 'where', 'why', 'when', 'who']):
        faq_results = faq_store.similarity_search(query, k=k)
        if faq_results:
            return faq_results

    # Error/problem words → prioritize Discourse
    if any(word in query_lower for word in ['error', 'fail', 'problem', 'issue', 'broken']):
        return discourse_store.similarity_search(query, k=k)

    # Tutorial/learn words → prioritize GTN
    if any(word in query_lower for word in ['tutorial', 'learn', 'guide', 'how-to']):
        return gtn_store.similarity_search(query, k=k)

    # Default: search all and merge
    return merged_search(query, k=k)
```

## RAG Prompt Templates

### Template 1: Hierarchical Context

```python
def build_hierarchical_context(query: str):
    """Build context with clear source hierarchy"""
    faq_results = faq_store.similarity_search(query, k=2)
    gtn_results = gtn_store.similarity_search(query, k=3)
    discourse_results = discourse_store.similarity_search(query, k=3)

    context = []

    if faq_results:
        context.append("=== Quick Answer (FAQ) ===")
        for doc in faq_results:
            context.append(f"Q: {doc.metadata['question']}")
            context.append(doc.page_content)
            context.append("")

    if gtn_results:
        context.append("=== Official Documentation ===")
        for doc in gtn_results:
            context.append(f"Tutorial: {doc.metadata['title']}")
            context.append(doc.page_content[:500] + "...")
            context.append("")

    if discourse_results:
        context.append("=== Community Discussion ===")
        for doc in discourse_results:
            context.append(f"Topic: {doc.metadata['title']}")
            context.append(f"URL: {doc.metadata['url']}")
            context.append(doc.page_content[:300] + "...")
            context.append("")

    return "\n".join(context)

# Usage in RAG prompt
prompt = f"""Based on the following Galaxy Project documentation, answer the user's question.

{build_hierarchical_context(query)}

User Question: {query}

Answer:"""
```

### Template 2: Source Attribution

```python
def build_attributed_context(query: str, k: int = 8):
    """Build context with clear source attribution"""
    results = merged_search(query, k=k)

    context_parts = []
    for i, (doc, score, source) in enumerate(results, 1):
        title = doc.metadata.get('title', doc.metadata.get('question', 'Untitled'))

        if source == 'faq':
            source_label = "FAQ"
        elif source == 'tutorial':
            source_label = f"Tutorial: {doc.metadata.get('topic', 'General')}"
        else:
            source_label = f"Community Discussion ({doc.metadata.get('reply_count', 0)} replies)"

        context_parts.append(f"[{i}] {source_label}: {title}")
        context_parts.append(doc.page_content[:400])
        context_parts.append("")

    return "\n".join(context_parts)
```

### Template 3: Multi-Perspective

```python
def build_multiperspective_context(query: str):
    """Provide different perspectives on the same topic"""
    return {
        'official': gtn_store.similarity_search(query, k=2),
        'community': discourse_store.similarity_search(query, k=2),
        'quick_answer': faq_store.similarity_search(query, k=1)
    }

# Usage
perspectives = build_multiperspective_context(query)

prompt = f"""Answer the question using multiple perspectives from Galaxy documentation:

Official Documentation Perspective:
{perspectives['official'][0].page_content if perspectives['official'] else 'Not found'}

Community Experience Perspective:
{perspectives['community'][0].page_content if perspectives['community'] else 'Not found'}

Quick Answer:
{perspectives['quick_answer'][0].page_content if perspectives['quick_answer'] else 'Not found'}

User Question: {query}

Synthesize these perspectives into a comprehensive answer:"""
```

## Performance Characteristics

### Search Performance

| Collection | Documents | Chunks | Avg Query Time | Memory Usage |
|------------|-----------|--------|----------------|--------------|
| galaxy_faqs | 25 | 62 | <10ms | ~5 MB |
| iwc_workflows | 105 | 772 | ~20ms | ~20 MB |
| gtn_tutorials | 168 | 9,295 | ~50ms | ~100 MB |
| discourse_posts | 5,986 | 70,516 | ~200ms | ~2 GB |

### Optimization Tips

1. **Cache Frequent Queries**: Store common query results
2. **Limit k**: Use smaller k values (5-10) for faster results
3. **Filter by Metadata**: Use Chroma's metadata filtering for faster searches
4. **Lazy Loading**: Only load collections you need
5. **Batch Queries**: Process multiple queries together

## Metadata Filtering

### Filter by Category (FAQ)

```python
results = faq_store.similarity_search(
    "workflows",
    k=5,
    filter={"category": "For Individual Learners"}
)
```

### Filter by Topic (GTN)

```python
results = gtn_store.similarity_search(
    "assembly",
    k=5,
    filter={"topic": "assembly"}
)
```

### Filter by Date (Discourse)

```python
results = discourse_store.similarity_search(
    "installation",
    k=5,
    filter={"created_at": {"$gte": "2024-01-01"}}
)
```

### Filter by Engagement (Discourse)

```python
# Only highly-viewed topics
results = discourse_store.similarity_search(
    "error",
    k=5,
    filter={"views": {"$gte": 100}}
)

# Only topics with replies
results = discourse_store.similarity_search(
    "problem",
    k=5,
    filter={"reply_count": {"$gte": 2}}
)
```

## Complete Integration Example

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LMStudioEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize embeddings
embeddings = LMStudioEmbeddings(
    base_url="http://localhost:1234/v1",
    model="text-embedding-nomic-embed-text-v2-moe"
)

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Load collections
faq_store = Chroma(
    collection_name="galaxy_faqs",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

gtn_store = Chroma(
    collection_name="gtn_tutorials",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

discourse_store = Chroma(
    collection_name="discourse_posts",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

iwc_store = Chroma(
    collection_name="iwc_workflows",
    embedding_function=embeddings,
    persist_directory="./data/embeddings"
)

# Create RAG chain
def answer_question(query: str):
    """Answer question using all available sources"""

    # Retrieve from all sources
    faq_docs = faq_store.similarity_search(query, k=2)
    iwc_docs = iwc_store.similarity_search(query, k=2)
    gtn_docs = gtn_store.similarity_search(query, k=3)
    discourse_docs = discourse_store.similarity_search(query, k=3)

    # Build context
    context_parts = []

    if faq_docs:
        context_parts.append("Quick Answers:")
        for doc in faq_docs:
            context_parts.append(f"- {doc.page_content}")

    if iwc_docs:
        context_parts.append("\nRelevant Workflows:")
        for doc in iwc_docs:
            context_parts.append(f"- {doc.metadata['workflow_name']}: {doc.page_content[:300]}")

    if gtn_docs:
        context_parts.append("\nOfficial Tutorials:")
        for doc in gtn_docs:
            context_parts.append(f"- {doc.metadata['title']}: {doc.page_content[:300]}")

    if discourse_docs:
        context_parts.append("\nCommunity Discussions:")
        for doc in discourse_docs:
            context_parts.append(f"- {doc.metadata['title']}: {doc.page_content[:300]}")

    context = "\n".join(context_parts)

    # Generate answer
    prompt = f"""Based on the following Galaxy Project documentation, provide a comprehensive answer.

{context}

Question: {query}

Answer (be specific and cite sources when relevant):"""

    response = llm.invoke(prompt)

    # Add source URLs
    sources = []
    for doc in iwc_docs[:2]:
        sources.append(f"Workflow: {doc.metadata['workflow_name']} ({doc.metadata['url']})")
    for doc in gtn_docs[:2]:
        sources.append(f"Tutorial: {doc.metadata['title']} ({doc.metadata['source']})")
    for doc in discourse_docs[:2]:
        sources.append(f"Discussion: {doc.metadata['url']}")

    return {
        'answer': response.content,
        'sources': sources
    }

# Usage
result = answer_question("How do I create a workflow in Galaxy?")
print(result['answer'])
print("\nSources:")
for source in result['sources']:
    print(f"- {source}")
```

## Maintenance & Updates

### Check for Updates

```bash
# Check if new data is available
make status

# Update all collections (incremental - only processes changes)
make update-all
```

### Update Schedule

- **Daily**: Run `make update-all` to catch new forum posts
- **Weekly**: Check `make status` for statistics
- **Monthly**: Force rebuild to catch edited content

### Monitoring

```python
from chromadb import PersistentClient

def get_collection_stats():
    """Get current collection statistics"""
    client = PersistentClient(path="./data/embeddings")

    stats = {}
    for coll_name in ["galaxy_faqs", "gtn_tutorials", "discourse_posts"]:
        coll = client.get_collection(coll_name)
        stats[coll_name] = coll.count()

    return stats

# Check if collections are healthy
stats = get_collection_stats()
print(f"FAQ chunks: {stats['galaxy_faqs']}")
print(f"Tutorial chunks: {stats['gtn_tutorials']}")
print(f"Discourse chunks: {stats['discourse_posts']}")
```

## Embedding Model Requirements

**Important**: Your embedding model must match the one used to create the collections.

**Current Model**: `text-embedding-nomic-embed-text-v2-moe` (768 dimensions)

**Alternatives**:
- OpenAI: `text-embedding-3-small` (requires re-embedding)
- SambaNova: `e5-mistral-7B-instruct` (requires re-embedding)

To use a different model, run:
```bash
# Change EMBEDDINGS_MODEL in .env
# Then rebuild all collections
make update-all --force-rebuild
```

## Directory Structure

```
data/
├── embeddings/
│   ├── chroma.sqlite3              # Vector database
│   ├── .discourse_tracking.json    # Incremental tracking
│   ├── .gtn_tracking.json
│   └── .faq_tracking.json
```

**Integration Requirements**:
- Read access to `data/embeddings/` directory
- LM Studio (or compatible) running on `http://localhost:1234`
- Same embedding model as used during creation

## Support

- **Documentation**: [README.md](README.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Discourse Pipeline**: [DISCOURSE_PIPELINE.md](DISCOURSE_PIPELINE.md)
- **Issues**: File on GitHub repository

## Version

- System Version: 2.0
- Last Updated: October 20, 2025
- Total Chunks: 79,873
- Collections: 3 (FAQ, GTN, Discourse)
