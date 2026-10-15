# Task: Build an Open Jev Exemplar Database from `shenjunhao/OpenJevData-140k`

## 1. Objective

Implement a Python command-line tool that downloads and converts the Hugging Face dataset:

https://huggingface.co/datasets/shenjunhao/OpenJevData-140k

into the format required by the existing pjev exemplar database pipeline.

The final output must be a SQLite + sqlite-vec exemplar database compatible with the existing R0/R1 retrieval infrastructure.

The project already provides:

- SQLite3
- sqlite-vec
- LZ4
- xxHash64 from llama.cpp
- Nomic embedding model at `models/nomic-embed-text-v1.5.gguf`
- Native C++ exemplar database builder, `pjevfilldb`
- R0 retrieval evaluation code
- R1 retrieved 1-shot evaluation code, if already implemented

Python package management must use **uv**, which is already installed.

The main priority is compatibility with the existing database format.

Do not build an independent, incompatible Python implementation of the embedding and database stack.

## 2. Recommended architecture

Use a two-stage pipeline:

```text
Hugging Face
shenjunhao/OpenJevData-140k
        |
        v
Python + Hugging Face datasets
        |
        v
Source schema validation
        |
        v
Open Jev-compatible JSONL
        |
        v
Existing native pjevfilldb
        |
        +-- xxHash64 identifiers
        +-- Canonical embedding text
        +-- Nomic GGUF embedding
        +-- LZ4 compression
        +-- SQLite relational tables
        +-- sqlite-vec vectors
        |
        v
models/examplar.db
```

The Python script is responsible for dataset conversion and coordinating the native builder.

The native builder remains responsible for embedding inference and the actual SQLite schema.

Avoid duplicating the C++ implementation of:

- xxHash64
- LZ4 compression
- Nomic GGUF inference
- Canonical embedding text generation
- sqlite-vec table creation
- Database versioning

If the existing native builder cannot accept the converted records, make a small, reusable extension to the native pipeline.

Do not silently discard source information simply to satisfy the old schema.

## 3. Inspect the repository first

Before implementing the converter:

1. Locate the existing `pjevfilldb` executable and source.
2. Inspect its accepted JSONL schema.
3. Inspect the actual exemplar SQLite schema.
4. Check the schema version.
5. Identify the embedding formatter.
6. Identify the LZ4 payload format.
7. Identify xxHash64 ID generation.
8. Identify R0/R1 reconstruction requirements.
9. Check existing command-line options and rebuild behavior.
10. Identify the actual current database filename.

Previous repository discussions used `models/exsamplar.db`, whereas this task requests `models/examplar.db`.

Treat `models/examplar.db` as the requested new output path.

Do not silently overwrite either existing database.

Do not change any existing filename globally.

Use explicit CLI paths, and report any naming discrepancies discovered.

## 4. Python environment with uv

Implement the converter as a Python script.

Suggested location:

`tools/convert_openjev140k.py`

Use uv for dependency management.

Suggested dependencies:

- `datasets`
- `huggingface_hub`
- `pyarrow`, if required by the installed datasets version

Prefer existing repository dependency declarations.

If the repository uses `pyproject.toml`, integrate dependencies there without disrupting unrelated environments.

Otherwise, a standalone uv-compatible script is acceptable.

Example setup:

```bash
uv add datasets huggingface_hub pyarrow
```

Alternatively, for a standalone script:

```bash
uv run --with datasets \
       --with huggingface_hub \
       --with pyarrow \
       tools/convert_openjev140k.py --help
```

Use the repository's established uv conventions.

Avoid pip-based installation instructions unless absolutely necessary.

Do not introduce PyTorch, Transformers, or a second GGUF runtime merely to convert records.

## 5. Source dataset

Load:

```python
from datasets import load_dataset

dataset = load_dataset(
    "shenjunhao/OpenJevData-140k",
    revision=resolved_revision,
)
```

The published dataset has:

- `SFT`: 100,345 records
- `RL`: 46,393 records
- Total: 146,738 records

These are source counts, not guaranteed final converted counts.

The dataset contains:

- English and Chinese content
- Three task kinds
- Hard and soft answer distributions
- Two to 77 candidate options
- Structured option objects
- Score values for score tasks
- Long natural-language states

### Inspect the actual loaded schema

Before implementing field mapping, print the discovered features and inspect representative rows from both SFT and RL.

Do not assume the source schema is exactly the same as the original Open Jev JSONL schema.

## 6. Default inclusion policy

The existing pjev exemplar system is primarily designed for English task examples.

Default to:

```text
language = English
source_split = SFT
answer_kind = hard
```

However, the source dataset does not necessarily provide a reliable per-row language field.

Implement a documented language filtering policy.

Do not rely only on the language of the question, because an English question may accompany a Chinese state.

Prefer explicit language metadata if actually available.

If language detection is necessary, use a deterministic, lightweight policy and classify uncertain mixed-language records separately.

Do not claim perfect English identification.

The converter must provide an option to retain both languages, but English-only must be the default.

### Split selection

Support:

```text
--splits SFT
--splits SFT,RL
--splits RL
```

Default to `SFT`.

Preserve the original split name as source provenance during conversion.

SFT and RL are training-stage partitions, not validation and test partitions.

Do not rename RL as validation.

## 7. Source schema mapping

Map the following source fields.

| Source | Open Jev output |
|---|---|
| `state` | `state` |
| `question` | `question` |
| `options` | `options` |
| `task` | `kind` |
| `category` | `metadata.domain` |
| `score_values` | `metadata.score_values` |
| `answer.probabilities` | `target` |
| `answer.kind` | `metadata.answer_kind` |
| SFT / RL | `metadata.source_split` |

Preserve the full meaning of model-visible source fields.

Do not invent a policy family, group structure, or counterfactual relationship.

## 8. State representation

The source `state` is primarily a natural-language string.

Preserve it exactly.

Do not automatically convert the text into an artificial structured state.

Do not summarize it.

Do not translate Chinese text into English.

Do not remove evidence, numeric values, or relevant context.

The converted record may contain:

```json
{
  "state": "Original source state text"
}
```

If the existing native builder requires state to be a JSON object, investigate its actual contract.

Prefer supporting valid string states in the native formatter.

Do not wrap the string in a redundant object unless that is the cleanest compatible representation.

If wrapping is required, use a documented consistent format, such as:

```json
{
  "state": {
    "text": "Original source state text"
  }
}
```

Ensure that R0 and R1 use the same state serialization.

## 9. Option conversion

Source options are objects:

```json
[
  {
    "id": "o0",
    "name": "Candidate A",
    "criteria": null
  },
  {
    "id": "o1",
    "name": "Candidate B",
    "criteria": "Must satisfy requirement X"
  }
]
```

The existing exemplar pipeline may expect an ordered array of strings.

Convert each option into a deterministic string while preserving all model-visible semantics.

Recommended rendering:

```text
Candidate A
```

If criteria are present:

```text
Candidate B
Criteria: Must satisfy requirement X
```

Never discard non-null criteria.

Do not include the source option ID if it is only a positional identifier with no semantic meaning.

However, preserve the original positional order.

Do not sort the options.

Do not convert option IDs into the final A/B/C labels inside the stored option text.

The existing prompt renderer should control answer-letter mapping.

Ensure deterministic rendering for nested or complex criteria.

## 10. Target distributions

This dataset differs from the original one-hot-only examples.

The source supports:

```text
answer.kind = hard
answer.kind = soft
```

### Hard targets

Hard answers use a one-hot probability vector.

Example:

```json
{
  "kind": "hard",
  "probabilities": [0.0, 1.0, 0.0]
}
```

Convert to:

```json
{
  "target": [0.0, 1.0, 0.0]
}
```

The existing native builder may derive:

```text
answer_index = 1
```

Ensure that the original option ordering is preserved.

### Soft targets

Soft answers contain an actual probability distribution.

Example:

```json
{
  "kind": "soft",
  "probabilities": [0.2, 0.5, 0.3]
}
```

Do not replace this with:

```json
{
  "target": [0.0, 1.0, 0.0]
}
```

Argmax conversion destroys the probability target's meaning.

Support an explicit option:

```text
--answer-kinds hard
--answer-kinds hard,soft
```

Default to `hard`.

For soft records, only allow inclusion if the existing database schema and downstream consumers can preserve their true probability distributions.

If soft-target support is not available, skip those records with explicit statistics.

Do not treat soft targets as ordinary hard-label few-shot demonstrations.

Future R1 experiments involving soft answers require a separately defined demonstration and evaluation policy.

### Validation

For every accepted target:

- Target length equals option count.
- Every probability is finite.
- Every probability is nonnegative.
- Sum is approximately 1.0.
- Hard targets are exactly one-hot within the documented numerical tolerance.

Preserve numeric precision sufficient for the original soft distributions.

## 11. Task kinds

The source field `task` may be:

- `choice`
- `noul`
- `score`

Map it to:

```json
{
  "kind": "score"
}
```

Do not confuse `task` with `answer.kind`.

These are separate concepts.

For example:

```text
task = choice
answer.kind = soft
```

is valid.

### Score tasks

Preserve `score_values` when present.

The number of levels must match the intended score scale.

Do not assume every score task has five levels.

Validate that score values align with the corresponding options.

Use the native builder's existing score-level representation.

The sqlite-vec search filters must continue to support:

```text
kind = score
number_of_levels = N
```

## 12. Identifier generation

The source dataset may not supply stable Open Jev-style textual IDs.

Generate deterministic IDs during conversion.

Suggested identifier scheme:

```text
openjev140k/SFT/00000000
openjev140k/SFT/00000001
openjev140k/RL/00000000
```

Use the original source order and split.

Do not generate random UUIDs.

Do not use unstable Python `hash()` values.

The native `pjevfilldb` must convert the generated string identifiers into xxHash64 values.

Do not independently implement xxHash64 in Python unless necessary for validation.

### Group IDs

Do not invent semantic group relationships.

If no original group ID exists, initially assign a unique group ID per source record.

Use deterministic, source-specific identifiers.

The same source row must receive the same ID on repeated conversions from the same pinned dataset revision.

### Deduplication

A deterministic per-record ID is not the same as content deduplication.

If both SFT and RL are included, optionally detect exact duplicate semantic examples using a canonical content fingerprint.

At minimum, report duplicates.

Do not silently merge examples with different targets.

## 13. Metadata policy

The optimized exemplar database should remain compact.

Do not store all source metadata in SQLite.

Retain only data needed by retrieval and few-shot reconstruction.

Useful fields include:

- Kind
- Domain/category
- Source split, where needed
- Score values
- Hard/soft answer kind, if needed
- Group identifier

The database may retain task-family metadata if actually provided.

Do not fabricate `template_id`, `scenario_family`, or `source_instance_id`.

Do not add:

- `semantic_context_sha256`
- Complete original provenance JSON
- Arbitrary unused metadata
- Long textual record IDs

Keep full conversion provenance in a sidecar manifest file instead of repeating it in every database row.

The native database builder should continue to use xxHash64 IDs and LZ4-compressed payloads.

## 14. Canonical embedding representation

Reuse the existing pjev embedding pipeline.

The stored training exemplars must use the Nomic document prefix:

```text
search_document:
```

Do not independently generate embeddings from Python.

Let the native builder:

1. Parse converted Open Jev records.
2. Build canonical embedding text.
3. Run the Nomic GGUF model.
4. Store float32 vectors.
5. Compress selected payloads using LZ4.
6. Insert metadata and vectors into SQLite.

### Natural-language states

The existing canonical formatter may have been designed around structured policy states.

Extend it if necessary to support natural-language state strings.

Preserve relevant information from:

- State
- Question
- Option names
- Option criteria
- Task kind
- Score definitions

Do not embed the correct answer or target distribution.

Do not embed incidental source IDs.

### Context-length limits

The source contains long states.

Inspect the Nomic embedding model's supported token limit.

Never silently truncate oversized records.

Use a deterministic policy and report:

- Oversized record count
- Actual handling
- Records skipped
- Maximum observed input length

Do not silently change existing database embedding semantics to accommodate longer source examples.

If a new canonical embedding-text format is required, version it explicitly and ensure R0/R1 query embeddings use the matching format.

## 15. Output paths

Suggested defaults:

```text
Intermediate JSONL:
  datasets/openjev140k-exemplars.jsol

Final database:
  models/examplar.db

Conversion manifest:
  datasets/openjev140k-exemplars.manifest.json
```

Support explicit CLI overrides:

```text
--output-jsonl PATH
--output-db PATH
```

The script must not overwrite an existing database unless the user explicitly supplies a rebuild or overwrite option.

Prefer building into a temporary database and atomically replacing the destination after verification.

## 16. CLI

Suggested invocation:

```bash
uv run tools/convert_openjev140k.py \
  --splits SFT \
  --language en \
  --answer-kinds hard \
  --output-jsonl datasets/openjev140k-exemplars.jsol \
  --output-db models/examplar.db \
  --model models/nomic-embed-text-v1.5.gguf
```

Implement:

```text
--splits SFT|RL|SFT,RL
--language en|all
--answer-kinds hard|hard,soft
--output-jsonl PATH
--output-db PATH
--model PATH
--revision REVISION
--limit N
--validate-only
--rebuild
--verbose
--help
```

The script should:

1. Load the dataset.
2. Inspect the actual schema.
3. Convert records.
4. Validate output.
5. Write JSONL.
6. Produce a conversion manifest.
7. Invoke the native `pjevfilldb` builder.
8. Verify the resulting database.
9. Print a final summary.

Allow users to skip the database-building phase for debugging.

Do not automatically install a new embedding backend.

## 17. Dataset reproducibility

Resolve and record the Hugging Face dataset revision.

Store the commit SHA in the conversion manifest.

Record:

- Dataset repository
- Resolved revision
- Selected splits
- Language filter
- Answer-kind filter
- Conversion format version
- Number of input records
- Number of output records
- Skip reasons
- Output JSONL checksum
- Native builder version, if available
- Embedding model fingerprint
- Database schema version

The database's internal metadata must remain consistent with the embedding model actually used.

## 18. Validation and statistics

Report counts by:

- Source split
- Task kind
- Answer kind
- Source category
- Detected language
- Number of options
- Number of score levels

Also report:

- Invalid source records
- Malformed answer distributions
- Missing required fields
- Duplicate IDs
- Duplicate semantic records
- Unsupported soft targets
- Unsupported languages
- Embedding context-length failures
- Successfully inserted exemplars

Do not assume all 146,738 records should appear in the final database when filtering is enabled.

Explain every meaningful reduction in record count.

## 19. Integration tests

Add Python unit tests and native integration checks.

### Converter tests

Verify:

- SFT record parsing
- RL record parsing
- Choice, noul, and score mapping
- Hard target preservation
- Soft target preservation or explicit exclusion
- Option order preservation
- Option criteria preservation
- Score values
- String state handling
- Unicode and Chinese state handling
- Stable IDs
- Deterministic JSONL output
- Correct language filtering
- Dataset revision recording

### Database integration

Verify:

- Existing native builder accepts the converted JSONL.
- Database uses INTEGER hashed IDs.
- LZ4 payloads round-trip correctly.
- Exemplar vector count matches the number of inserted rows.
- Internal example rowids match vector rowids.
- Kind filtering works.
- Score-level filtering works.
- Retrieved hard-label examples reconstruct correctly.
- Source question and options are preserved.
- The database opens using the existing R0 reader.

### Retrieval smoke test

Run a small real-model smoke test.

Use queries independent from the records inserted into the database.

Verify:

- Correct vector dimensions
- Finite distances
- Valid retrieved example IDs
- Correct kind filtering
- Correct score-level filtering
- Successful LZ4 decompression

Do not assess retrieval quality by querying a record and retrieving the identical stored record.

Do not claim benchmark improvements from a smoke test.

## 20. Evaluation contamination

This is especially important.

OpenJevData-140k contains examples derived from multiple public benchmark sources.

Do not assume that it is disjoint from:

- Open Jev validation
- JevBench
- CLEAN-50K
- Other external evaluation corpora

Before using the new database for R0/R1 evaluation, inspect available source provenance and test for overlap where possible.

At minimum, support exact text-content duplicate checks against the intended evaluation set.

If stronger provenance is unavailable, state the limitation.

Do not claim improved generalization based solely on retrieval from a potentially overlapping corpus.

Keep the existing exemplar database available for baseline comparisons.

## 21. Storage and performance

The final database should use the existing optimized schema:

- xxHash64 INTEGER identifiers
- LZ4-compressed state payloads
- LZ4-compressed options
- LZ4-compressed embedding text, if retained
- Uncompressed float32 embeddings
- Native sqlite-vec kind filtering
- Native score-level filtering

Report:

- Number of examples
- Database size
- Embedding dimension
- Mean embedding throughput
- Total build time
- Duplicate count
- Compression statistics, where available

Do not recompute embeddings unnecessarily when an already compatible partial database can safely be resumed.

However, never mix incompatible embedding models or formatting versions in one vector table.

## 22. Out of scope

Do not implement:

- A new SQLite schema unrelated to pjev
- A new Python GGUF inference engine
- A separate Python vector database
- A different embedding model
- R2 similarity gating
- Adaptive 1-shot/2-shot inference
- kNN answer voting
- Fine-tuning
- Changes to production pjev inference
- Arbitrary soft-target argmax conversion
- Dataset translation
- Benchmark threshold optimization

The task is dataset conversion and database construction.

## 23. Acceptance criteria

The task is complete when:

1. uv runs the converter successfully.
2. The dataset is loaded from a pinned revision.
3. SFT and RL can be selected independently.
4. All three task kinds are correctly mapped.
5. Hard answer distributions remain one-hot.
6. Soft distributions are preserved or explicitly excluded.
7. Option criteria are not lost.
8. Natural-language states are preserved.
9. English filtering works according to a documented policy.
10. Converted JSONL is accepted by the native builder.
11. The final database uses xxHash64 and LZ4.
12. Embeddings use the existing Nomic model integration.
13. The existing R0 reader can query the resulting database.
14. Vector rowid relationships are valid.
15. Repeated conversion is deterministic.
16. The existing database is not overwritten without explicit authorization.
17. Conversion and integration tests pass.
18. The final report includes actual counts, database size, and limitations.

## 24. Deliverables

Provide:

- `tools/convert_openjev140k.py`
- Python tests
- uv dependency configuration
- Converted JSONL
- Conversion manifest
- Final SQLite database
- Build and usage instructions
- Validation statistics
- R0 retrieval smoke-test results
- Known limitations

Include the exact CLI command required to reproduce the final database.

Do not claim successful execution unless the corresponding command and tests actually ran.

## Final instruction

Prioritize fidelity to the source dataset and compatibility with the existing pjev exemplar infrastructure.

Use Python and uv for downloading, validating, and converting the Hugging Face dataset.

Use the existing native C++ pipeline for xxHash64, LZ4, Nomic embedding generation, and SQLite/sqlite-vec storage.

Do not introduce a parallel embedding or database implementation.

**Do not turn soft probability distributions into hard labels, do not discard option criteria, and do not silently truncate long states.**