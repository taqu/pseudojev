# Task: Replace Text-Based Exemplar IDs with xxHash64 SQLite INTEGER Keys

## 1. Objective

Optimize the existing exemplar database by replacing long text-based `id` and `group_id` values with 64-bit hashes.

The database is:

`models/exsamplar.db`

The current implementation stores Open Jev identifiers such as:

```text
community-diversity-v2/20260921/support/refund_window/0/1/score
```

These identifiers are unnecessarily long for runtime retrieval and few-shot inference.

Their original textual meanings are not required by pjev at runtime.

Use xxHash64 to convert them into fixed-size 64-bit identifiers and store the resulting values as native SQLite `INTEGER` columns.

The repository already includes an xxHash implementation originating from llama.cpp.

**Reuse the existing xxHash implementation. Do not introduce another hashing dependency.**

The goal is to reduce database size, improve identifier storage efficiency, and simplify runtime handling without changing embedding vectors or retrieval behavior.

## 2. Scope

Update the following components as necessary:

- `tools/pjevfilldb`
- Exemplar database schema
- Database reader and writer
- Existing R0 retrieval tool
- Existing or in-progress R1 retrieval and few-shot code
- Database tests
- Schema documentation

Do not modify the main inference model or unrelated pjev behavior.

Do not modify:

- Embedding generation
- Canonical embedding text
- Cosine distance calculations
- Task-kind filtering
- Score-level filtering
- Few-shot prompt semantics
- LZ4 compression format

This task is a database identity representation optimization only.

## 3. Inspect the existing implementation

Before making changes:

1. Inspect the current SQLite schema.
2. Identify all TEXT identifier columns.
3. Identify where `id` and `group_id` are generated or parsed.
4. Identify how exemplar IDs are used in vector-table relationships.
5. Identify which R0/R1 diagnostics depend on textual identifiers.
6. Locate the existing llama.cpp-derived xxHash implementation.
7. Confirm the available xxHash64 function signature and seed handling.
8. Inspect the current schema version and database rebuild mechanism.

Do not assume the current schema exactly matches previous design proposals.

Preserve the actual repository architecture and coding conventions.

## 4. Identifier hashing

Implement a small reusable helper using the existing xxHash64 implementation.

Suggested interface:

```cpp
int64_t hash_exemplar_id(std::string_view text);
```

The function must:

- Use xxHash64.
- Use a fixed seed of zero.
- Hash the complete original identifier.
- Hash the raw UTF-8 bytes without normalization.
- Return a signed 64-bit value suitable for SQLite.
- Produce deterministic results across executions and platforms.
- Avoid undefined behavior.

Do not use:

- `std::hash`
- Random seeds
- Process-specific hash functions
- Truncated 32-bit hashes
- Cryptographic hashing libraries
- New external hash dependencies

### Unsigned-to-signed conversion

xxHash64 produces an unsigned 64-bit value.

SQLite `INTEGER` stores signed 64-bit integers.

Preserve the exact 64-bit bit pattern when converting the hash.

For C++20, `std::bit_cast<int64_t>(hash)` is suitable on the supported conventional 64-bit representation.

Otherwise, use a well-defined, tested equivalent compatible with the repository's C++ standard.

Do not rely on implementation-defined out-of-range unsigned-to-signed casts.

Never convert the hash into a decimal TEXT value before storing it.

Use SQLite's 64-bit integer binding interface:

```cpp
sqlite3_bind_int64(stmt, index, hashed_id);
```

Read the value using:

```cpp
sqlite3_column_int64(stmt, column);
```

Negative INTEGER values are valid and expected for hashes whose highest bit is set.

Do not take the absolute value or mask away the sign bit.

### Hash input rules

Hash only the original identifier bytes.

For example:

```cpp
auto id_hash = hash_exemplar_id(record["id"]);
auto group_hash = hash_exemplar_id(record["group_id"]);
```

Do not hash the JSON serialization of a string.

Do not add arbitrary prefixes or suffixes.

Do not include the task kind, split, state, or answer in the hash input.

The same original `group_id` must always generate the same numeric group ID.

## 5. Database schema changes

Replace:

```sql
id TEXT NOT NULL UNIQUE,
group_id TEXT NOT NULL
```

with:

```sql
id INTEGER NOT NULL UNIQUE,
group_id INTEGER NOT NULL
```

If the database contains a `groups` table, change its identifier to:

```sql
CREATE TABLE groups (
    group_id INTEGER PRIMARY KEY
    -- Preserve only other necessary columns.
);
```

Keep the remaining schema as close to the current version as possible.

Do not add columns containing the original ID strings.

Do not store the original text in compressed metadata as a fallback.

### Important: Keep the internal rowid

If the current schema uses:

```sql
rowid INTEGER PRIMARY KEY
```

as the internal example key, preserve that design.

Do not automatically replace the internal rowid with the xxHash64 identifier.

The intended mapping remains:

```text
examples.rowid
    ==
exemplar_vectors.rowid
```

The hashed `examples.id` is a logical identifier.

The SQLite rowid is an internal storage and vector-join identifier.

Keep these responsibilities separate.

### Recommended structure

```text
examples
  rowid             INTEGER PRIMARY KEY
  id                INTEGER UNIQUE NOT NULL
  group_id          INTEGER NOT NULL
  kind              TEXT NOT NULL
  number_of_levels  INTEGER
  question          TEXT
  state_lz4         BLOB
  options_lz4       BLOB
  embedding_text_lz4 BLOB
  ...
```

Preserve the existing LZ4 payload representation.

Preserve the existing sqlite-vec virtual table.

## 6. Collision policy

Hash collisions are considered acceptable for this application.

The database is an approximate exemplar retrieval resource, not an authoritative identity store.

Do not implement:

- Collision-resolution tables
- Original-text ID fallback columns
- Secondary cryptographic hashes
- Collision retry loops
- Collision suffixes
- Collision-specific schema extensions

A collision may cause an exemplar to be omitted or an identity comparison to treat two records as equivalent.

This limitation is acceptable.

However, database corruption and inconsistent vector mappings are not acceptable.

### Example ID collision

If two distinct source records map to the same hashed `id`, treat the later record as a duplicate according to the existing duplicate-handling policy.

Do not overwrite the earlier stored exemplar with conflicting content.

Prefer `INSERT ... ON CONFLICT DO NOTHING` or an equivalent explicit policy.

Do not use `INSERT OR REPLACE` if it may delete and recreate a row or break the vector rowid relationship.

Count the skipped records in ingestion statistics.

No expensive collision investigation is required.

### Group ID collision

If two different source groups produce the same hashed `group_id`, they may be treated as the same group.

This is acceptable for the intended retrieval use case.

However, a group collision must not cause an exemplar's vector to be attached to the wrong example row.

Use the existing foreign-key and transaction handling.

The probability of xxHash64 collisions is sufficiently small for the expected database size.

Do not spend implementation effort attempting to make the identity scheme collision-free.

## 7. Database construction

Modify `pjevfilldb` so that textual identifiers are converted immediately after JSONL parsing.

Suggested flow:

```text
Read JSONL record
       |
       v
Extract original id and group_id
       |
       v
Compute xxHash64
       |
       v
Discard textual identifiers
       |
       v
Build canonical embedding text
       |
       v
Generate embedding
       |
       v
Insert example and vector
```

In practice, perform duplicate detection before expensive embedding inference whenever possible.

Suggested order:

1. Parse record.
2. Validate required fields.
3. Compute hashed IDs.
4. Check for an existing hashed example ID.
5. Skip duplicates according to policy.
6. Generate canonical embedding text.
7. Compute embedding.
8. Insert the example and vector atomically.

Do not retain large source ID strings beyond the processing scope unless required by temporary parser objects.

## 8. Read-only retrieval

Update R0 and R1 database readers to use integer identifiers.

Examples:

```cpp
int64_t example_id;
int64_t group_id;
```

Use 64-bit integer SQL operations and bindings.

Do not convert hash values back into strings for internal comparisons.

Group comparisons should use integer equality:

```cpp
bool same_group =
    query_group_hash == retrieved_group_hash;
```

The evaluation query's original textual group ID must be hashed with the same function and seed.

### R0 diagnostics

Continue supporting:

- Same-group detection
- Retrieval result identification
- Retrieval frequencies
- Duplicate retrieval diagnostics
- Source-family analysis where other retained metadata permits it

Use numeric IDs in machine-readable output.

If a human-readable report displays hashed IDs, hexadecimal formatting is acceptable for readability.

Hexadecimal output is only for presentation.

The actual database storage must remain INTEGER.

### R1 inference

R1 must continue to:

- Retrieve Top-1 eligible exemplar
- Resolve it using the stable internal rowid
- Decompress LZ4 payloads
- Reconstruct the demonstration
- Run the existing answer-scoring logic

The numerical hash is not part of the few-shot prompt.

Do not insert hashed IDs into prompts unless the existing task content genuinely requires them.

## 9. Existing database migration or rebuild

The existing `models/exsamplar.db` may contain textual identifiers.

Implement a safe conversion strategy.

### Preferred approach: Preserve existing embeddings

If technically practical, migrate identifiers without recomputing embeddings.

The migration should:

1. Open the existing database.
2. Create a new database with the updated schema.
3. Read original example rows.
4. Hash `id` and `group_id`.
5. Copy existing LZ4 payloads without decompressing or recompressing them.
6. Preserve the existing internal example rowids.
7. Copy existing embedding vectors and metadata.
8. Validate rowid relationships.
9. Validate vector counts.
10. Validate representative KNN results.
11. Replace the existing database only after successful validation.

Do not modify the original database in place before the new database has passed validation.

Preserve a recoverable copy of the old database.

If the installed sqlite-vec version does not support the intended vector-copy procedure, use the supported mechanism rather than manipulating virtual-table shadow tables directly.

Do not copy sqlite-vec internal shadow tables manually.

### Alternative: Rebuild from JSONL

If migration is excessively complex or unsafe, rebuild using the updated `pjevfilldb`.

Document the additional embedding computation cost.

Do not silently rebuild when a migration was requested.

### Schema version

Increment the database schema version.

Older database readers must not silently interpret integer hashes as text identifiers or vice versa.

Reject unsupported schema versions with meaningful error messages.

## 10. Performance considerations

The expected benefit is reduced storage overhead from long text identifiers and related SQLite index entries.

However, note that the internal example rowid and vector-rowid mapping already provide efficient numeric joins.

Therefore, do not claim that hashing automatically accelerates vector search.

Measure the actual difference.

Pay attention to:

- Main table size
- ID index size
- Group ID index size
- Overall database size
- Database construction time
- Lookup performance

Do not introduce a per-row hash lookup during vector search when a direct rowid lookup is already available.

The retrieved vector rowid should remain the primary mechanism for resolving the selected exemplar.

## 11. Required tests

### Hash implementation

Test:

- Empty string
- Short identifier
- Long Open Jev identifier
- UTF-8 input
- Repeated hashing
- Stable zero-seed behavior
- Values with the highest bit set
- Signed SQLite INTEGER round-trip
- Different ID strings
- Same group ID across different records

Include known xxHash64 test vectors compatible with the implementation used by the repository.

### Database ingestion

Test:

- Correct hashed example ID
- Correct hashed group ID
- Duplicate IDs
- Simulated hash collision handling
- Multiple examples sharing one group
- Correct rowid mapping
- Correct transaction rollback
- Existing database rebuild or migration

A simulated collision can use a test-specific hash function or controlled fixture.

Do not attempt to discover an actual xxHash64 collision.

### R0

Verify:

- Existing Top-K retrieval still works.
- Kind filtering is unchanged.
- Score-level filtering is unchanged.
- Same-group comparisons work with hashed IDs.
- Diagnostic output contains valid numeric identifiers.
- Vector distances and rankings remain unchanged after a lossless migration.

### R1

Verify:

- Top-1 retrieval still works.
- LZ4 payload reconstruction works.
- Correct exemplar is selected by vector rowid.
- Correct answer is reconstructed.
- Few-shot prompt content is unchanged.
- S1/R1 accuracy calculations are unaffected by ID representation.

## 12. Storage measurements

Measure database size before and after conversion.

Report:

| Metric | Before | After |
|---|---|---|
| Total DB size | measured | measured |
| Example table size | measured | measured |
| Identifier index size | measured | measured |
| Vector count | measured | measured |
| Example count | measured | measured |
| Retrieval result parity | baseline | verified |

Use SQLite's available database statistics facilities where practical.

Run checkpoint and compaction procedures consistently before measuring file sizes.

Do not attribute changes caused by unrelated VACUUM or compression operations solely to ID hashing.

The vector storage size should remain essentially unchanged.

## 13. CLI compatibility

Preserve existing `pjevfilldb` CLI behavior.

For example:

```bash
tools/pjevfilldb \
  --input datasets/train.jsol \
  --output models/exsamplar.db \
  --model models/nomic-embed-text-v1.5.gguf \
  --rebuild
```

Adjust the executable name and flags to match the actual repository implementation.

Do not introduce unnecessary new flags.

Do not require the runtime user to understand or configure the hash seed.

The fixed hash convention is part of the database schema.

## 14. Out of scope

Do not implement:

- A new hashing algorithm
- Cryptographic ID hashing
- Collision-resolution infrastructure
- Additional metadata compression
- New embedding representations
- New vector indexes
- Retrieval reranking
- Changes to R1 prompt design
- Adaptive retrieval
- Changes to the pjev model
- Changes to task answer scoring

Only optimize identifier storage and ensure downstream compatibility.

## 15. Acceptance criteria

The task is complete when:

1. `id` is stored as SQLite INTEGER.
2. `group_id` is stored as SQLite INTEGER.
3. Both values are generated with the existing xxHash64 implementation.
4. The original textual IDs are not stored in the optimized database.
5. The original internal rowid-to-vector mapping is preserved.
6. LZ4 payloads remain compatible.
7. R0 retrieval works correctly.
8. R1 retrieval and few-shot reconstruction work correctly, if R1 is already implemented.
9. Hashing is deterministic.
10. Signed 64-bit SQLite storage works correctly.
11. Existing database migration or rebuilding is supported safely.
12. Relevant tests pass.
13. Actual storage savings are measured.
14. The new schema version is documented.

## 16. Deliverables

After implementation, provide:

- Changed files
- Updated database schema
- Hash function used
- Hash seed
- Signed 64-bit conversion method
- Migration or rebuild command
- Database size before and after conversion
- R0 retrieval parity test results
- R1 compatibility test results, if applicable
- Test execution results
- Known limitations

Do not claim tests passed unless they were actually executed.

## Final instruction

This is a storage optimization.

Keep the implementation minimal.

Use the repository's existing xxHash64 implementation, replace long text identifiers with native SQLite INTEGER values, preserve vector rowid mapping, and maintain the current retrieval behavior.

**Hash collisions are acceptable, but inconsistent database state, broken joins, or incorrect vector-to-exemplar mapping are not.**