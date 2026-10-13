# Task: Optimize the Exemplar Database Schema and Add LZ4 Compression

## 1. Objective

Refactor the existing Open Jev exemplar database builder to reduce database size and eliminate unnecessary stored metadata.

The existing database is:

`models/exsamplar.db`

LZ4 has already been installed.

The refactoring must:

1. Remove metadata not required for R0 retrieval diagnostics or future R1 few-shot inference.
2. Compress large text and JSON payloads with LZ4.
3. Keep indexed and frequently filtered columns uncompressed.
4. Preserve the ability to reconstruct complete few-shot examples.
5. Maintain compatibility with R0 retrieval.
6. Keep embedding vectors uncompressed.
7. Provide a safe database rebuild or migration procedure.

Do not implement R1 inference or adaptive retrieval in this task.

## 2. Inspect the existing implementation

Before modifying code, inspect:

- `tools/pjevfilldb`
- Existing SQLite schema
- sqlite-vec integration
- Existing LZ4 integration
- R0 retrieval code
- R0 report generation
- Embedding canonical text builder
- Existing database versioning

Identify which database columns are actually read by the current R0 implementation.

Do not blindly apply an assumed schema.

Prefer deleting unused columns rather than preserving them for hypothetical future features.

## 3. Minimal database schema

Retain the following fields in the main example table, unless repository inspection demonstrates that a field is unnecessary.

### Identifiers and filtering

- `rowid` — INTEGER PRIMARY KEY
- `id` — TEXT UNIQUE
- `group_id` — TEXT
- `kind` — TEXT
- `number_of_levels` — INTEGER, nullable
- `domain` — TEXT, nullable
- `family` — TEXT, nullable
- `scenario_family` — TEXT, nullable
- `template_id` — TEXT, nullable
- `source_instance_id` — TEXT, nullable

Keep these as native SQLite columns.

They may be needed for filtering, grouping, or retrieval diagnostics.

### Question and answer

- `question` — TEXT
- `answer_index` — INTEGER, nullable

The original option ordering must remain recoverable.

For valid one-hot targets, the target vector can be reconstructed from `answer_index` and the number of options.

If non-one-hot targets exist, preserve them in a separate compressed payload rather than silently discarding information.

Do not assume all task kinds always have one-hot targets without checking the dataset and current API contract.

### Compressed payloads

Use SQLite BLOB columns:

- `state_lz4` — BLOB
- `options_lz4` — BLOB
- `embedding_text_lz4` — BLOB

Store compressed UTF-8 representations.

Use deterministic JSON serialization for structured values.

Store additional compressed target data only if required for lossless reconstruction.

### Fields to remove

Remove unless the current R0 implementation has a demonstrated requirement:

- `semantic_context_sha256`
- `metadata_json`
- `counterfactual_json`
- `embedding_model` duplicated per example
- `embedding_version` duplicated per example
- `language`, if the database is strictly English-only
- `split`, if the database is strictly train-only
- Unused provenance subfields
- Redundant answer text that can be reconstructed from options

Keep model and format information once in the database metadata table rather than on every example.

Do not retain the full metadata object merely for archival purposes.

The source JSONL remains the authoritative source for reconstructing omitted provenance.

## 4. LZ4 compression

Use the installed native LZ4 library.

Prefer the standard LZ4 Frame API:

- `LZ4F_compressFrameBound`
- `LZ4F_compressFrame`
- `LZ4F_decompress`

Store each compressed payload as an independent LZ4 frame.

Advantages:

- Self-describing frame format
- Independent decompression
- Explicit error detection
- No custom raw-block size protocol required

Implement reusable functions equivalent to:

```cpp
std::vector<std::byte> lz4_compress(
    std::string_view input
);

std::string lz4_decompress(
    std::span<const std::byte> compressed
);
```

Adapt the interfaces to the C++ standard already used by the project.

### Compression requirements

- Support arbitrary UTF-8.
- Preserve exact original bytes.
- Support empty strings.
- Detect malformed or truncated frames.
- Avoid unbounded decompression allocations.
- Enforce a configurable maximum decompressed size.
- Verify all LZ4 error return values.
- Bind compressed payloads to SQLite using BLOB parameters.
- Never store compressed binary data as TEXT.

Use one consistent compression format across the database.

Do not compress small identifiers, numeric metadata, or vector embeddings.

### Small payload optimization

LZ4 framing has overhead.

Do not assume that compression always reduces storage.

Measure compressed sizes for:

- State JSON
- Options JSON
- Embedding text

Implementing a raw-versus-compressed fallback is optional.

If implemented, it must use an explicit, documented encoding marker so decompression is unambiguous.

Prefer simplicity and correctness over marginal savings.

## 5. Vector storage

Keep the existing sqlite-vec implementation.

Do not LZ4-compress the embedding vectors.

Preserve:

- float32 vectors
- Actual model dimensionality
- Cosine distance
- Stable example rowid mapping
- Kind-specific filtering
- Score-level filtering

The vector table should contain only the vector and metadata required for vector search.

Do not duplicate long text payloads inside the vec0 virtual table.

Use the relational example table to retrieve and decompress payloads only for selected exemplars.

## 6. Retrieval behavior

R0 retrieval must continue to work.

The intended query sequence is:

1. Generate query embedding.
2. Search sqlite-vec with kind and score-level filters.
3. Obtain nearest exemplar row IDs and distances.
4. Join with the example table.
5. Retrieve any required compressed payloads.
6. Decompress only the payloads needed for the requested report or downstream use.

Do not decompress every example before searching.

Do not deserialize state JSON for every vector candidate.

For R0 summary statistics, metadata and distances should normally be sufficient.

For human-readable retrieval reports, decompress the selected examples.

For future R1 inference, decompress only the exemplar or exemplars selected for the prompt.

## 7. Database metadata and versioning

Maintain a database-level metadata table containing:

- Schema version
- Embedding model identity and fingerprint
- Embedding dimension
- Embedding format version
- Cosine distance configuration
- Embedding normalization policy
- Compression format: `lz4-frame`
- Payload serialization version

Increment the schema version.

Database readers must reject unsupported schema versions with a clear error.

Do not silently interpret compressed columns as legacy plain-text columns.

## 8. Rebuild and migration

The existing database may use the previous uncompressed schema.

Support one of the following safe approaches:

**Preferred: Rebuild from the original training JSONL**

Use the existing `pjevfilldb` ingestion pipeline with the optimized schema.

If the existing embedding model and canonical text format remain unchanged, avoid unnecessary embedding recomputation where a safe vector-preserving migration is possible.

**Alternative: Explicit schema migration**

Copy existing data into a new database, compress supported payloads, and preserve embeddings.

Requirements:

- Do not modify the original database in place before validation.
- Preserve IDs and vector rowid relationships.
- Verify row counts.
- Verify vector counts.
- Verify compressed payload round trips.
- Verify representative KNN queries.
- Replace the old database only after successful validation and with a recoverable backup.

Do not silently delete or overwrite the existing database.

## 9. Required tests

Add or update tests for:

### Compression

- Small text
- Large text
- Empty text
- UTF-8 strings
- Structured JSON
- Truncated LZ4 data
- Corrupted LZ4 data
- Maximum decompressed size enforcement
- Exact compression/decompression round trips

### Database

- Correct compressed payload storage
- Correct payload reconstruction
- No redundant metadata storage
- Correct one-hot answer reconstruction
- Preservation of non-one-hot targets, if present
- Stable example/vector rowid mapping
- Database reopen
- Rebuild safety

### Retrieval

- Choice-only retrieval
- Noul-only retrieval
- Score-level filtering
- Top-K ordering
- Human-readable R0 reports
- Retrieval without decompressing all records

## 10. Benchmark storage efficiency

Report database sizes before and after optimization.

Include:

- Total SQLite database size
- Vector storage size, where measurable
- Average uncompressed state size
- Average compressed state size
- Average uncompressed options size
- Average compressed options size
- Average uncompressed embedding text size
- Average compressed embedding text size
- Overall compression ratio
- Mean and P95 decompression latency

Measure actual file size after a consistent checkpoint and database compaction procedure where appropriate.

Do not compare a compacted new database with a fragmented old database without noting the difference.

The purpose is to demonstrate real storage savings.

## 11. Compatibility with R0

Update R0 only where necessary to read the optimized schema.

Do not change:

- Query embedding construction
- Similarity calculations
- Mandatory kind filtering
- Score-level filtering
- Retrieval evaluation logic
- R0 output semantics

For a fixed query set, compare retrieval before and after migration.

The retrieved exemplar IDs, distances, and ordering should remain equivalent if the vectors and filters are unchanged.

If source metadata has been intentionally removed, update reports to omit those diagnostics rather than inventing replacement values.

Preserve group/template/source-instance diagnostics where the corresponding retained columns support them.

## 12. Acceptance criteria

The task is complete when:

1. The optimized schema is implemented.
2. Unnecessary metadata is no longer stored.
3. Large text and JSON payloads use LZ4.
4. Compression is lossless.
5. Embedding vectors remain uncompressed.
6. Kind and score-level filters still work.
7. Few-shot examples can be fully reconstructed.
8. R0 continues to run.
9. Database creation and reopening work.
10. Migration or rebuild is safe.
11. Unit and integration tests pass.
12. Before/after size measurements are reported.

## 13. Deliverables

Report:

- Changed files
- New database schema
- Removed columns and justification
- LZ4 implementation details
- Database build or migration command
- Compatibility test results
- Before/after database sizes
- Compression ratios
- Known limitations

Do not implement new inference features.

**The goal is a compact, efficient exemplar database that stores only what is necessary for retrieval and few-shot reconstruction, while preserving the existing R0 retrieval behavior.**