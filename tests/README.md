# Galaxy Documentation Search Tests

Comprehensive test suite for validating search quality across all Galaxy documentation collections.

## Test Files

### `test_search_quality.py`
Tests fundamental search functionality and metadata correctness:
- Collection loading and metadata structure
- Metadata type validation
- Search accuracy for different query types
- Cross-collection search
- Retrieval strategies (prioritized, merged, weighted)
- Edge cases and error handling
- Collection size validation

### `test_search_scenarios.py`
Real-world search scenarios and use cases:
- Finding bioinformatics workflows (RNA-seq, variant calling, etc.)
- Tutorial discovery (beginner, analysis-specific)
- Quick answers from FAQs
- Problem-solving with community discussions
- Specialized analyses (COVID-19, structural biology, proteomics)
- Multi-source retrieval
- Metadata utility (citations, authors, URLs)
- Search relevance validation
- Complex multi-part queries

## Prerequisites

1. **Embeddings must be built first**:
```bash
# Run all pipelines to create embeddings
make update-all
```

2. **LM Studio running** with `text-embedding-nomic-embed-text-v2-moe` model:
```bash
# Start LM Studio on http://localhost:1234
```

3. **Environment variables set** in `.env`:
```bash
EMBEDDINGS_API_KEY=local-model
EMBEDDINGS_BASE_URL=http://127.0.0.1:1234/v1
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_search_quality.py -v
pytest tests/test_search_scenarios.py -v
```

### Run specific test class
```bash
pytest tests/test_search_quality.py::TestIWCCollection -v
pytest tests/test_search_scenarios.py::TestBioinformaticsWorkflows -v
```

### Run specific test
```bash
pytest tests/test_search_quality.py::TestIWCCollection::test_iwc_analysis_type_search -v
```

### Run tests matching a pattern
```bash
# All RNA-seq related tests
pytest tests/ -k "rnaseq or RNA" -v

# All workflow tests
pytest tests/ -k "workflow" -v

# All metadata tests
pytest tests/ -k "metadata" -v
```

### Run with example output
```bash
# Includes the example results printer
pytest tests/test_search_scenarios.py::test_print_example_results -v -s
```

## Test Organization

### By Collection

**FAQ Tests**:
- `TestFAQCollection` - FAQ-specific search quality
- `TestQuickAnswers` - Getting quick answers

**IWC Tests**:
- `TestIWCCollection` - IWC workflow search quality
- `TestBioinformaticsWorkflows` - Finding analysis workflows
- `TestSpecializedAnalyses` - COVID-19, structural biology, proteomics

**GTN Tests**:
- `TestGTNCollection` - GTN tutorial search quality
- `TestTutorialDiscovery` - Finding relevant tutorials

**Discourse Tests**:
- `TestDiscourseCollection` - Discourse post search quality
- `TestProblemSolving` - Finding community solutions

### By Functionality

**Search Quality**:
- `TestSearchRelevance` - Results are actually relevant
- `TestComplexQueries` - Multi-part queries

**Integration**:
- `TestCrossCollectionSearch` - Searching multiple collections
- `TestMultiSourceRetrieval` - Getting complementary info

**Metadata**:
- `TestMetadataUtility` - Citations, authors, URLs

**Strategies**:
- `TestRetrievalStrategies` - Different search approaches

## Common Test Patterns

### Parameterized Tests
Many tests use `@pytest.mark.parametrize` to test multiple queries:

```python
@pytest.mark.parametrize("query,expected_topic", [
    ("RNA-seq analysis", "transcriptomics"),
    ("variant calling", "variant-calling"),
])
def test_analysis_type_search(self, iwc_store, query, expected_topic):
    results = iwc_store.similarity_search(query, k=5)
    # assertions...
```

### Fixtures
Tests use shared fixtures for efficiency:
- `embeddings` - Shared embeddings instance
- `faq_store`, `iwc_store`, `gtn_store`, `discourse_store` - Collection stores
- `all_stores` - Dictionary of all stores

## Expected Test Results

With properly built embeddings, you should see:
- ✅ All collections load successfully
- ✅ Metadata is complete and correctly typed
- ✅ Searches return relevant results
- ✅ Different query types work (questions, analysis types, problems)
- ✅ Multi-collection searches find complementary information
- ✅ Specialized searches (COVID-19, proteomics, etc.) find appropriate workflows

## Troubleshooting

### Tests fail with "collection not found"
```bash
# Make sure embeddings are built
make update-all

# Check collections exist
ls -la data/embeddings/
```

### Tests fail with "EMBEDDINGS_API_KEY not set"
```bash
# Check .env file exists and has correct values
cat .env | grep EMBEDDINGS
```

### Tests fail with connection errors
```bash
# Make sure LM Studio is running
curl http://127.0.0.1:1234/v1/models

# Should return list of models including text-embedding-nomic-embed-text-v2-moe
```

### Tests pass but relevance seems poor
```bash
# Check if correct embedding model is used
# Model must match what was used to create embeddings
pytest tests/test_search_quality.py::TestIWCCollection::test_iwc_metadata_structure -v
```

## Adding New Tests

### Test a new analysis type
Add to `test_search_scenarios.py`:

```python
def test_new_analysis_type(self, all_stores):
    """Test finding workflows for new analysis."""
    iwc = all_stores['iwc']
    results = iwc.similarity_search("your analysis type", k=5)

    assert len(results) > 0
    # Add specific assertions for your analysis
```

### Test a new retrieval pattern
Add to `test_search_quality.py`:

```python
def test_new_strategy(self, faq_store, iwc_store, gtn_store, discourse_store):
    """Test new retrieval strategy."""
    # Implement and test your strategy
```

## CI/CD Integration

Tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run search tests
  run: |
    # Start LM Studio or use cloud embeddings
    pytest tests/ --junitxml=test-results.xml
```

## Performance

Test suite performance:
- **Full suite**: ~2-3 minutes (with local embeddings)
- **Per test**: ~0.5-2 seconds (depends on collection size)
- **With cloud embeddings**: Add API latency

Tips for faster tests:
- Use `pytest -n auto` for parallel execution (requires pytest-xdist)
- Run specific test classes during development
- Use `--maxfail=1` to stop on first failure

## Contributing

When adding tests:
1. Use descriptive test names (`test_find_rnaseq_workflows` not `test_search_1`)
2. Add docstrings explaining what's being tested
3. Use parameterize for similar test cases
4. Keep assertions specific and meaningful
5. Add to appropriate test class by functionality
