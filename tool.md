# Task: Implement `tools/pjevfilldb` — Open Jev Exemplar Database Builder

## 1. Objective

Implement a new command-line tool, `tools/pjevfilldb`, that converts the Open Jev training dataset into a persistent SQLite exemplar database with precomputed semantic embeddings.

This tool is a prerequisite for AR0 (Retrieval Sanity) in the adaptive retrieval roadmap.

The resulting database will eventually support retrieval-augmented few-shot inference in pjev.

**This task is limited to database construction. Do not implement AR0 evaluation, runtime retrieval integration, adaptive gating, or few-shot inference yet.**

## 2. Repository and dependencies

The following dependencies and files are already available:

- SQLite3
- sqlite-vec
- Embedding model: `models/nomic-embed-text-v1.5.gguf`
- Training dataset: `datasets/train.jsol`
- New tool location: `tools/pjevfilldb`

Before implementing anything:

1. Inspect the existing repository structure and build configuration.
2. Identify the JSON library currently used by pjev.
3. Identify the existing llama.cpp integration, if present.
4. Identify how SQLite3 and sqlite-vec have been integrated.
5. Follow the repository's existing C++ conventions, error-handling patterns, and build system.
6. Reuse existing abstractions where practical.

Do not introduce Python, an external embedding service, or an additional database engine.

Prefer a self-contained native C++ implementation using the dependencies already installed.

Avoid unnecessary architectural changes to the existing pjev runtime.

## 3. Input format

The input is a JSON Lines file.

Each line contains one complete Open Jev example.

Important fields include:

```json
{
  "group_id": "...",
  "id": "...",
  "kind": "score",
  "metadata": {
    "domain": "support",
    "family": "policy",
    "language": "en",
    "scenario_family": "support/refund_window",
    "template_id": "...",
    "source_instance_id": "...",
    "semantic_context_sha256": "...",
    "score_values": [0, 1, 2, 3, 4],
    "counterfactual": null
  },
  "options": [
    "Exactly 0 requirements are established.",
    "Exactly 1 requirement is established."
  ],
  "question": "...",
  "source": "...",
  "split": "train",
  "state": {},
  "target": [0.0, 1.0]
}
```

This is an illustrative structure, not a complete sample record.

The actual dataset may contain different state schemas, option counts, task kinds, and metadata fields.

Do not hard-code assumptions based on the refund-window example.

The `state` field is arbitrary structured JSON, not necessarily a string.

Process the file incrementally, one record at a time, rather than loading the entire dataset into memory.

## 4. Supported task types

The initial implementation must support these Open Jev kinds:

- `choice`
- `noul`
- `score`

The dataset uses the property `kind`, whereas the Jev API calls the equivalent concept `type`.

Treat these as the same semantic discriminator.

Use `kind` in the database to match the dataset.

All kinds may reside in one SQLite database and one vector table.

However, they must have logically separate retrieval spaces.

A future query for `score` must not return `choice` or `noul` exemplars.

The vector schema must support filtering by kind during KNN retrieval.

### Additional score constraint

For `score`, store:

- `number_of_levels`
- `score_values`, when available
- Original option ordering

For example, `score_values: [0,1,2,3,4]` represents five score levels.

Future score retrieval must be able to require an exact match on `number_of_levels`.

Do not assume all score tasks have four or five levels.

## 5. Database schema

Use a normalized relational structure for metadata and original examples, plus a separate sqlite-vec virtual table for embeddings.

Recommended logical tables:

### `groups`

Store shared group-level identifiers.

Suggested columns:

- `group_id` TEXT PRIMARY KEY
- `source` TEXT
- `scenario_family` TEXT

Do not assume all examples within a group have identical states.

### `examples`

Each input JSONL record must correspond to exactly one example row.

Suggested columns:

- `rowid` INTEGER PRIMARY KEY
- `id` TEXT UNIQUE NOT NULL
- `group_id` TEXT NOT NULL
- `kind` TEXT NOT NULL
- `state_json` TEXT NOT NULL
- `question` TEXT NOT NULL
- `options_json` TEXT NOT NULL
- `target_json` TEXT NOT NULL
- `answer_index` INTEGER NULL
- `answer_text` TEXT NULL
- `number_of_levels` INTEGER NULL
- `score_values_json` TEXT NULL
- `domain` TEXT NULL
- `family` TEXT NULL
- `language` TEXT
- `scenario_family` TEXT
- `template_id` TEXT
- `source_instance_id` TEXT
- `semantic_context_sha256` TEXT
- `counterfactual_json` TEXT NULL
- `metadata_json` TEXT
- `source` TEXT
- `split` TEXT NOT NULL
- `embedding_text` TEXT NOT NULL
- `embedding_format_version` INTEGER NOT NULL

Preserve the complete original state, options, target, and metadata without losing information.

Unknown optional metadata fields must not cause otherwise valid records to fail.

For one-hot classification targets, derive `answer_index` and `answer_text` from the target and ordered options.

For targets that cannot safely be interpreted as one-hot, preserve the original target and leave derived answer columns NULL.

Do not silently convert non-one-hot targets into classification labels.

### `exemplar_vectors`

Use sqlite-vec's `vec0` virtual table.

The vector table must contain:

- Embedding vector
- Kind metadata suitable for KNN filtering
- Number-of-levels metadata suitable for score filtering

Use cosine distance.

The standard nomic-embed-text-v1.5 dimension is 768. Verify the actual model output dimension before creating or populating the vector table.

Use a stable mapping:

`exemplar_vectors.rowid == examples.rowid`

This mapping must remain consistent across inserts, restarts, and rebuilds.

Check the sqlite-vec version installed in this repository and use syntax supported by that version.

Prefer metadata filters that participate in the KNN search rather than filtering only after obtaining global nearest neighbors.

If the installed sqlite-vec version has restrictions on metadata columns or NULL values, implement and document a compatible representation without compromising exact kind and score-level filtering.

### `db_info`

Add a small metadata table containing:

- Schema version
- Embedding model identifier
- Embedding model fingerprint or checksum
- Embedding dimension
- Embedding text format version
- Distance metric
- Normalization policy
- Embedding runtime configuration relevant to reproducibility

The tool must reject incompatible updates to an existing database unless an explicit rebuild is requested.

## 6. Embedding model

Use:

`models/nomic-embed-text-v1.5.gguf`

Generate embeddings locally using the project's existing llama.cpp integration, if available.

Do not use a network embedding API.

For indexed examples, use the model's document task prefix:

`search_document:`

Future query embeddings will use:

`search_query:`

Implement the embedding generator as a reusable component so future retrieval code can use the same model configuration.

Requirements:

- Load the embedding model once per execution.
- Do not reload it for every record.
- Use the pooling mode appropriate for this GGUF model.
- Produce float32 embeddings.
- Use a consistent, documented L2 normalization policy.
- Validate embedding dimensions.
- Reject NaN, infinity, or invalid zero-norm vectors.
- Ensure document and future query embeddings use compatible configurations.
- Explicitly handle token-length limits.

Do not silently truncate input texts in a way that changes their meaning. Report unsupported oversized inputs, or use an explicitly documented deterministic policy.

CPU execution must work. Optional GPU acceleration may follow existing project conventions.

## 7. Canonical embedding representation

Do not embed raw JSON serialization directly.

Create a deterministic English text representation for semantic retrieval.

The canonical text must describe the task being performed, not its correct answer.

Build it using:

- Task kind
- Scenario
- Question
- Relevant state rules
- Evidence policy
- Numbered requirements, when present
- Selection rule, when applicable
- Relevant options or score meanings

Avoid introducing target information or derived correct answers into the embedding text.

### Important: structured state

The Open Jev state may contain:

- Candidate lists
- Candidate facts
- Three-valued evidence rules
- Numbered requirements
- Request parameters
- Selection rules
- Other task-specific fields

Handle these structures deterministically.

Do not build a formatter that only understands `support/refund_window`.

Implement a generic recursive formatter for arbitrary JSON, supplemented by specialized formatting for known semantic fields when useful.

A first-version canonical representation should emphasize task structure and relevant reasoning requirements.

For example:

```text
Task: score
Domain: support
Family: policy

Scenario:
Select an admissible refund handling plan.

Question:
Candidate P41. How many requirements are established?

Evidence policy:
Unknown evidence does not establish a requirement.

Requirements:
1. days_since_delivery <= 14
2. payment_settled equals true
3. NOT refund_already_paid
4. seal_intact OR defect_confirmed

Score meanings:
0: Exactly 0 requirements are established.
1: Exactly 1 requirement is established.
2: Exactly 2 requirements are established.
3: Exactly 3 requirements are established.
4: Exactly 4 requirements are established.
```

This is illustrative, not a mandatory universal template.

### Candidate data policy

Use a structure-oriented canonical representation as the initial default.

Preserve the meaning of the question and the identity of any referenced candidate or entity where needed.

Avoid embedding arbitrary candidate identifiers, long irrelevant candidate lists, or incidental numeric values unless they are necessary to understand the task.

However, the formatter must not accidentally erase distinctions between fundamentally different task types or operations.

Always preserve the complete original state separately in SQLite.

Make the formatter independently testable and version its output.

AR0 may later compare structure-oriented embeddings against embeddings containing all candidate facts.

Do not implement that experiment now.

## 8. Correct-answer handling

Preserve the original option order.

For a one-hot target:

```text
options = ["no", "yes"]
target  = [1.0, 0.0]
```

Store:

```text
answer_index = 0
answer_text  = "no"
```

For:

```text
options = ["K11", "item_82", "P72"]
target  = [0.0, 1.0, 0.0]
```

Store:

```text
answer_index = 1
answer_text  = "item_82"
```

Do not assume the answer is always A/B/C/D.

The future few-shot prompt builder will need to reconstruct option labels consistently with the target task's answer mapping.

Do not put the correct answer into the embedding representation.

## 9. Grouping and counterfactual relationships

Open Jev contains related examples identified by:

- `group_id`
- `source_instance_id`
- `template_id`
- `scenario_family`
- `semantic_context_sha256`
- `metadata.counterfactual`

These identifiers are important for future leakage analysis and retrieval diagnostics.

Preserve them.

Do not collapse related examples into one row.

Do not deduplicate examples merely because they share a group, source instance, or template.

Counterfactual variants may change individual facts while retaining the same group ID.

Each unique example ID must retain its own:

- State
- Question
- Options
- Target
- Embedding

Only exact repeated example IDs should be treated as potential duplicates.

If the same ID is encountered with conflicting content, report the conflict rather than silently overwriting it.

## 10. Train-only exemplar bank

By default, include only records satisfying:

```text
split == train
metadata.language == en
```

No validation, calibration, test, or OOD examples should enter the exemplar bank by default.

Reject or skip nonmatching records with explicit counters.

Preserve source-family and template metadata to support future benchmark-leakage analysis.

Do not implement evaluation-set retrieval or any operation that merges held-out examples into the train database.

## 11. CLI

Implement a command-line interface equivalent to:

```bash
tools/pjevfilldb \
  --input datasets/train.jsol \
  --output data/exemplars.db \
  --model models/nomic-embed-text-v1.5.gguf
```

Additional options:

```text
--rebuild
--limit N
--batch-size N
--verbose
--help
```

Expected semantics:

- `--rebuild`: explicitly recreate the generated database after confirmation through the flag itself.
- `--limit N`: process at most N eligible examples, useful for smoke tests.
- `--batch-size N`: configure bounded processing/commit batches.
- `--verbose`: enable detailed progress logging.
- `--help`: show usage information.

The tool should create the output parent directory if necessary, following repository conventions.

All failures must produce a nonzero exit status.

Do not automatically destroy an existing incompatible database.

## 12. Transaction and restart behavior

Database creation must be deterministic and restart-safe.

Requirements:

1. Use prepared statements and bound parameters.
2. Use transactions for insert batches.
3. Insert the example row and its corresponding vector atomically.
4. Enforce unique example IDs.
5. Support safe reruns without duplicating existing records.
6. Roll back partially completed batches after failures.
7. Detect incompatible existing database metadata.
8. Avoid losing a previously valid database when rebuild fails.

For `--rebuild`, prefer building a new database in a temporary file and replacing the old generated database only after successful completion and validation.

Use a consistent duplicate-handling policy and report how many records were inserted or skipped.

## 13. Logging and statistics

Display a useful final summary.

Example:

```text
Input file: datasets/train.jsol
Output DB: data/exemplars.db

Records read:              12000
Eligible records:          10000
Inserted:                  10000
Duplicates skipped:            0
Invalid records:               0

By kind:
  choice:                   3000
  noul:                     3000
  score:                    4000

Embedding dimension:         768
Embedding model: nomic-embed-text-v1.5
Embedding text version:        1

Elapsed time: ...
Database size: ...
```

These numbers are illustrative.

Actual counts must come from the processed data.

Report invalid records with line numbers and meaningful error messages.

For malformed input, follow an explicit, documented policy rather than silently losing records.

Do not log complete state payloads by default.

## 14. Unit and integration tests

Add tests using the repository's existing testing framework.

At minimum, cover the following cases.

### JSON parsing

- Valid `choice` record
- Valid `noul` record
- Valid `score` record
- Missing required field
- Optional metadata omitted
- Malformed JSONL record
- Unknown task kind
- Non-English record
- Non-train record

### Answer extraction

- One-hot targets
- Zero-based answer indexing
- Preservation of option order
- Invalid target length
- Non-one-hot target handling
- Score level extraction

### Canonical text

- Stable deterministic output
- Identical input produces identical text
- Different kinds produce different task representations
- Original JSON is not mutated
- Counterfactual state differences are preserved in stored data
- The correct answer never appears through target-derived information

### Database

- Fresh database creation
- Correct example-to-vector rowid mapping
- Foreign-key integrity
- Rerun without duplicate inserts
- Interrupted batch rollback
- Rebuild behavior
- Embedding model mismatch detection
- Embedding dimension mismatch detection

### Vector retrieval smoke test

Create a small database containing at least:

- One choice example
- One noul example
- Two score examples with different level counts

Execute actual sqlite-vec KNN queries.

Verify that:

- `choice` queries cannot retrieve `noul` examples.
- `noul` queries cannot retrieve `score` examples.
- Five-level score queries exclude four-level score examples.
- Returned example IDs resolve to the correct original rows.
- Distance values are finite.
- The stored vectors have the expected dimensions.

Use real embeddings for at least one small end-to-end integration test.

Do not use mocked vectors as the only proof that the embedding pipeline works.

## 15. Performance considerations

The tool is an offline preprocessing utility.

Correctness, reproducibility, and safe restart behavior are more important than maximum throughput.

Nevertheless:

- Load the embedding model once.
- Use batch embedding where supported.
- Keep memory usage bounded.
- Avoid repeated parsing of the same record.
- Use prepared SQL statements.
- Commit records in configurable batches.
- Report throughput.

Do not implement HNSW or any additional approximate-nearest-neighbor index.

The initial dataset should use sqlite-vec's supported exact vector-search capabilities.

## 16. Code organization

Prefer a small set of focused components, approximately:

```text
tools/pjevfilldb/
    main.cpp
    jsol_reader.*
    exemplar_record.*
    embedding_text.*
    embedding_model.*
    exemplar_database.*
```

These filenames are suggestions, not strict requirements.

Adapt the organization to the existing repository.

Keep these responsibilities separate:

- JSONL reading
- Record validation/extraction
- Canonical text generation
- Embedding inference
- SQLite persistence

The canonical text formatter should be reusable by the future query-side retrieval implementation.

Avoid building a new general-purpose framework.

## 17. Out of scope

Do not implement:

- AR0 retrieval-quality evaluation
- AR1 few-shot inference
- Runtime pjev API changes
- Confidence-based adaptive gates
- Similarity thresholds
- Exemplar reranking
- kNN answer voting
- Automatic prompt injection
- Benchmark accuracy experiments
- Retrieval-result caching
- HNSW or additional vector databases

The goal is a reliable exemplar database builder, not a complete retrieval system.

## 18. Acceptance criteria

The task is complete when:

1. `tools/pjevfilldb` builds using the existing project build system.
2. The CLI processes `datasets/train.jsol`.
3. Only English training examples enter the database by default.
4. `choice`, `noul`, and `score` are supported.
5. Each example is stored with its complete original data.
6. One embedding is stored per eligible example.
7. Embeddings are generated by the local GGUF model.
8. Kind-specific KNN filtering works.
9. Score-level filtering works.
10. Correct answers can be reconstructed from stored data.
11. Counterfactual and source-family metadata are preserved.
12. Running the tool again does not duplicate examples.
13. Build and runtime errors are handled cleanly.
14. Unit and integration tests pass.
15. Documentation includes build instructions, example CLI usage, database schema, and query examples.

## 19. Final deliverables

After implementation, provide:

- List of changed and added files
- Build instructions
- CLI usage examples
- Final database schema
- Example sqlite-vec KNN query with kind filtering
- Test results
- Brief explanation of embedding text design
- Known limitations or unresolved issues

Do not claim tests passed unless they were actually executed.

If a dependency or repository-specific API differs from the assumptions in this specification, inspect the installed implementation and adapt accordingly.

**Priority: deliver a correct, reproducible, well-tested `pjevfilldb` suitable as the foundation for AR0, without modifying the existing inference behavior.**