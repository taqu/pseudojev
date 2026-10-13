# Task: Implement a Python Converter for General CLEAN-50K to Open Jev R0 JSONL

## 1. Objective

Implement a Python command-line script that converts the Hugging Face **General CLEAN-50K** dataset into JSONL records compatible with the existing Open Jev R0 retrieval evaluation tool.

Source:

https://huggingface.co/datasets/samatv256/jev-decisions-v1

Dataset configuration:

`general-clean-50k`

The output must be compatible with the existing Open Jev record interface:

```json
{
  "group_id": "...",
  "id": "...",
  "kind": "choice",
  "metadata": {},
  "options": [],
  "question": "...",
  "source": "...",
  "split": "validation",
  "state": {},
  "target": []
}
```

The purpose is to evaluate whether the existing Nomic embedding and exemplar retrieval system generalizes to decision-making tasks from an external dataset.

**Do not modify or rebuild the existing exemplar database.**

Database path:

`models/exsamplar.db`

Do not implement retrieval or model inference in this task. Only implement dataset conversion, validation, and a small compatibility smoke test.

## 2. Understand the source dataset

Use:

```python
from datasets import load_dataset

dataset = load_dataset(
    "samatv256/jev-decisions-v1",
    "general-clean-50k",
    split="train",
)
```

The published dataset card states that General CLEAN-50K contains:

- 50,000 records
- Choice-only decision supervision
- State
- Fixed Choice question
- Finite ordered answer options
- Correct option index

The source is already privacy-sanitized.

Important: inspect the actual loaded dataset features and several representative records before implementing field mappings.

The full underlying Jev Decisions v1 dataset has a different, more extensive schema. Do not assume that its columns are identical to those of General CLEAN-50K.

Do not guess column names when the actual Parquet schema is available.

Print the discovered schema during development and implement conversion against the real fields.

Pin or record the Hugging Face dataset revision for reproducibility.

## 3. Repository inspection

Before implementing:

1. Inspect `tools/pjevfilldb`.
2. Inspect the existing R0 tool.
3. Identify the exact JSONL input expectations.
4. Inspect how `state`, `question`, `options`, and `target` are parsed.
5. Identify required metadata for kind filtering.
6. Check whether the R0 canonical formatter supports the proposed converted state.
7. Inspect how R0 interprets dataset `split`.

Reuse the existing data contract.

Do not modify the existing R0 implementation unless a compatibility problem requires a small, clearly documented change.

Do not alter R0 retrieval behavior.

## 4. Script location and CLI

Suggested location:

`tools/convert_clean50k.py`

Example invocation:

```bash
python tools/convert_clean50k.py \
  --output datasets/clean50k-r0.jsol
```

Support:

```text
--output PATH
--limit N
--revision REVISION
--validate-only
--strict
--verbose
--help
```

Optional:

```text
--input-parquet PATH
```

Allow conversion from locally downloaded Parquet files to avoid repeated downloads.

Use the Hugging Face `datasets` library for remote loading.

Use `pyarrow` if appropriate for local Parquet input.

Do not require a new C++ dependency.

Use UTF-8 output with one JSON object per line.

## 5. Record mapping

Convert each source record into an Open Jev-compatible record.

### 5.1 Kind

All CLEAN-50K examples are Choice tasks.

Set:

```json
"kind": "choice"
```

Do not generate `score` or `noul` records.

### 5.2 ID

Preserve the original source decision ID in metadata.

Generate a stable namespaced ID:

```text
clean50k/<original_decision_id>
```

If the original ID is unavailable, use a deterministic content hash rather than a random UUID.

Verify ID uniqueness.

### 5.3 Group ID

Prefer a stable source trajectory or task identity when that information is actually available in the CLEAN-50K schema.

Otherwise, use the unique converted record ID as `group_id`.

Do not infer trajectories from unrelated records or arbitrary ID prefixes.

Preserve the original source record ID separately.

Do not fabricate source-family or counterfactual relationships.

### 5.4 Question

Preserve the original Choice question exactly if provided by the source.

Do not replace it with a newly invented question.

If the source schema has no explicit question, use a documented constant Choice question only if it is consistent with the published dataset's intended task.

Example fallback:

```text
Which of the available actions should be selected
given the current state?
```

The fallback must never depend on the correct answer.

### 5.5 State

Preserve the source's model-visible state.

The output `state` must be a JSON object.

If the source state is a structured object, retain its semantic structure.

If source fields contain JSON-serialized strings, parse them where appropriate and safe.

Do not double-encode the state into an opaque JSON string unless required by the existing R0 interface.

Do not insert the correct option index, answer, target action, or hidden evaluation labels into the state.

Preserve:

- System instructions
- User goal
- Available history
- Relevant environment context

Only when these fields actually exist in the CLEAN-50K record.

Do not invent missing information.

Keep the history in its original order.

### 5.6 Options

The Open Jev interface expects:

```json
"options": [
  "option 0",
  "option 1",
  "option 2"
]
```

Convert each CLEAN-50K option into a deterministic string.

Preserve the original candidate order exactly.

Do not sort options alphabetically or by candidate name.

An option may contain:

- Action type
- Action name
- Description
- Parameter schema

Use only the symmetric, model-visible option fields supplied by CLEAN-50K.

Suggested rendering:

```text
Type: tool
Name: search_documents
Description: Search available documents.
Parameters: {"query":{"type":"string"}}
```

These are illustrative fields.

Use the actual source schema.

For nested parameter schemas, serialize deterministically with stable key ordering.

Do not include:

- Target arguments
- Hidden labels
- Candidate metadata not intended as model input
- Evaluation-only information

Ensure every output option is a valid nonempty string.

Avoid unnecessary verbose JSON formatting.

Preserve all available candidate distinctions necessary for interpreting the Choice task.

### 5.7 Target

Convert the correct option index into an Open Jev one-hot vector.

For example:

```text
Source:
  correct_option_index = 2
  options count = 4
```

Output:

```json
"target": [0.0, 0.0, 1.0, 0.0]
```

Requirements:

- Exactly one value must be 1.0.
- All other values must be 0.0.
- Target length must equal option count.
- Preserve the source's zero-based or one-based indexing semantics after verifying the actual schema.
- Reject invalid or ambiguous target indices.
- Never reorder options to make labels easier to process.

Do not infer the target from textual similarity, action names, or candidate descriptions.

Use the explicit source supervision field.

### 5.8 Split

The source CLEAN-50K split is `train`.

However, this converted dataset is intended to be used as an external query dataset for R0.

Preserve the true source split in metadata:

```json
"source_split": "train"
```

If the existing R0 reader requires `split = validation`, set:

```json
"split": "validation"
```

only as a compatibility alias.

Also record:

```json
"evaluation_role": "external_retrieval_queries"
```

Document clearly that the underlying examples remain source-training examples.

Do not describe this dataset as the official Open Jev validation set.

If R0 supports an explicit external query role without rewriting the split, prefer preserving the original split instead.

## 6. Metadata

Preserve useful provenance.

Suggested metadata:

```json
{
  "language": "en",
  "domain": "agent_decision",
  "family": "tool_choice",
  "dataset": "samatv256/jev-decisions-v1",
  "dataset_config": "general-clean-50k",
  "source_split": "train",
  "evaluation_role": "external_retrieval_queries",
  "original_id": "...",
  "dataset_revision": "...",
  "conversion_version": 1
}
```

Only hard-code values that are genuinely constant for this source.

Preserve additional source metadata when actually available, including source dataset and trajectory identity.

Do not invent a `template_id`, `scenario_family`, or `source_instance_id` merely to match the Open Jev schema.

These fields may be omitted or set to null if unknown.

The R0 report should be able to distinguish missing metadata from actual nonmatches.

## 7. Important: Prevent answer leakage

This is a critical requirement.

No information derived from the correct answer may appear in the model-visible fields.

Model-visible fields include:

- `state`
- `question`
- `options`

The correct answer must appear only in `target` and optional diagnostic metadata excluded from embedding generation.

The conversion must never:

- Insert the correct index into state.
- Mark the correct option.
- Put the selected action into question text.
- Add hidden target arguments to options.
- Move a correct-action payload into the state.
- Include source evaluation labels in the embedding input.

The source dataset explicitly distinguishes model-visible fields from target supervision.

Respect that distinction.

Implement tests for accidental answer leakage.

## 8. Preserve full text and option sets

CLEAN-50K includes some long contexts and large candidate sets.

Do not arbitrarily truncate:

- State
- History
- Option descriptions
- Parameter schemas
- Candidate lists

Do not impose an artificial four-option limit.

Do not drop examples just because they have more than eight options.

If downstream R0 or the embedding model has a context-length limitation, report it separately.

Do not silently truncate during conversion.

The converter should preserve the complete model-visible input so retrieval experiments can later compare truncation or summarization strategies.

## 9. JSONL output format

Each output line must be a complete Open Jev record.

Illustrative example:

```json
{
  "group_id": "clean50k/example123",
  "id": "clean50k/example123",
  "kind": "choice",
  "metadata": {
    "language": "en",
    "domain": "agent_decision",
    "family": "tool_choice",
    "dataset": "samatv256/jev-decisions-v1",
    "dataset_config": "general-clean-50k",
    "source_split": "train",
    "evaluation_role": "external_retrieval_queries",
    "original_id": "example123",
    "conversion_version": 1
  },
  "options": [
    "Name: search_documents\nDescription: Search documents",
    "Name: ask_user\nDescription: Request clarification",
    "Name: finish\nDescription: Finish the task"
  ],
  "question": "Which action should be selected?",
  "source": "samatv256/jev-decisions-v1/general-clean-50k",
  "split": "validation",
  "state": {
    "user_goal": "Find a relevant document",
    "history": []
  },
  "target": [1.0, 0.0, 0.0]
}
```

This is only an illustrative output example.

Do not assume these exact field names or values exist in the source.

The actual conversion must be derived from the inspected CLEAN-50K schema.

## 10. Validation

Validate every output record.

Required checks:

1. Valid JSON object.
2. Nonempty unique ID.
3. Nonempty group ID.
4. `kind == "choice"`.
5. `state` is a JSON object.
6. `question` is a nonempty string.
7. `options` is an array of strings.
8. At least two options.
9. `target` length equals options length.
10. Exactly one target element equals 1.0.
11. All remaining target elements equal 0.0.
12. Correct option index is within bounds.
13. No unexpected target-derived fields in model-visible input.
14. No loss of source option ordering.

The published subset already required distinct options, but revalidate after string conversion because different source options could accidentally render to identical strings.

If this happens, do not merge the options or silently remove the row. Record the problem explicitly.

In strict mode, stop on the first invalid record.

In non-strict mode, skip invalid records with warnings and counters.

Do not silently repair malformed labels.

## 11. Conversion statistics

Generate:

`datasets/clean50k-r0.stats.json`

Include:

- Source dataset and revision
- Total records read
- Successfully converted records
- Invalid records
- Skipped records by reason
- Unique ID count
- Number of option-count mismatches
- Target validation failures
- Distribution of option counts
- Mean and maximum option count
- State text length distribution
- Output JSONL size
- Conversion version

If all source records are valid and no intentional filter is applied, the expected output is 50,000 records.

Do not assert success merely because the script completed.

Verify the actual output line count.

## 12. Determinism

Given identical:

- Source dataset revision
- Source records
- Converter version
- Conversion settings

The output must be deterministic.

Requirements:

- Stable record ordering
- Stable option ordering
- Stable JSON serialization
- Stable candidate rendering
- No random IDs
- No timestamps embedded in individual records
- No unordered iteration affecting text output

Use UTF-8 and `ensure_ascii=False`.

Preserve Unicode characters.

Document the serialization policy.

## 13. Tests

Add Python unit tests covering:

### Source conversion

- Two-option Choice example
- Five-option Choice example
- More than 16 options
- Nested parameter schemas
- Multiline state text
- Empty history
- Missing optional metadata
- Null optional fields

### Target conversion

- First option correct
- Middle option correct
- Last option correct
- Invalid negative index
- Out-of-range index
- Noninteger index
- Ambiguous target

### Output validation

- JSONL round-trip
- Unique IDs
- Stable option ordering
- One-hot target validity
- Deterministic serialization
- No target leakage
- No accidental option collisions

### R0 compatibility

Run a small conversion sample and verify that the existing R0 input parser accepts it.

Use the real R0 parser or binary, not only a separately invented Python schema.

If R0 cannot process the converted state, identify the exact compatibility issue.

Do not claim compatibility from JSON syntax validation alone.

## 14. R0 smoke test

After conversion, run a small retrieval-only smoke test.

Suggested command:

```bash
tools/pjev-r0 \
  --db models/exsamplar.db \
  --input datasets/clean50k-r0.jsol \
  --model models/nomic-embed-text-v1.5.gguf \
  --kind choice \
  --top-k 5 \
  --limit 100 \
  --output results/r0-clean50k-smoke
```

Adjust flags to match the actual R0 CLI.

The smoke test must:

- Read the existing database without modifying it.
- Process converted Choice examples.
- Generate query embeddings.
- Retrieve Choice exemplars only.
- Produce valid retrieval diagnostics.
- Report embedding errors or token-limit problems.
- Never run the main pjev inference model.

If embedding fails because CLEAN-50K contexts are longer than the supported model input, report the affected count and distribution.

Do not silently drop or truncate the input to make the smoke test pass.

A separate explicit R0-side truncation or representation policy can be designed later.

## 15. Avoid misleading evaluation claims

General CLEAN-50K is a training subset of Jev Decisions v1.

Its use as a query source does not make it the official validation set of either Jev Decisions v1 or Open Jev.

This experiment should be labeled:

**R0 External Corpus Retrieval Sanity Check**

Before reporting cross-corpus generalization, investigate possible overlaps between the source corpus and the exemplar database.

At minimum:

- Check exact record IDs where meaningful.
- Check any shared source identifiers.
- Check exact or normalized model-visible text hashes where possible.
- Report that source identity and hashes are not necessarily comparable across independently normalized datasets.

Do not claim proven absence of overlap when the necessary provenance is unavailable.

Do not use this converted file as an additional training exemplar bank during the same R0 experiment.

## 16. Scope exclusions

Do not:

- Modify `models/exsamplar.db`
- Insert CLEAN-50K into the exemplar bank
- Implement new embeddings for stored exemplars
- Implement R1
- Run Bonsai or the main pjev inference model
- Perform few-shot prompt construction
- Implement adaptive retrieval
- Optimize similarity thresholds
- Evaluate prediction accuracy
- Download the full 12M-record dataset unnecessarily

Only download or load `general-clean-50k` and any metadata required for provenance.

## 17. Deliverables

Provide:

1. `tools/convert_clean50k.py`
2. Python tests
3. `datasets/clean50k-r0.jsol`
4. `datasets/clean50k-r0.stats.json`
5. Usage documentation
6. R0 compatibility smoke-test results

If the repository does not normally commit generated datasets, follow its existing conventions and exclude the generated 50K JSONL from version control.

At completion, report:

- Discovered source schema
- Exact field mappings
- Dataset revision
- Conversion count
- Validation failures
- Sample converted record
- File size
- Test results
- R0 smoke-test result
- Any incompatibilities or context-length limitations

Do not report tests as passed unless they were actually executed.

## 18. Acceptance criteria

The task is complete when:

- The official General CLEAN-50K configuration can be loaded.
- All valid source records are converted.
- The output follows the actual R0 input contract.
- The Choice kind is preserved.
- Candidate order is preserved.
- Correct indices are accurately converted to one-hot targets.
- No answer information leaks into model-visible fields.
- Original source provenance is retained.
- Conversion is deterministic.
- Validation statistics are produced.
- A real R0 retrieval smoke test succeeds, or a precise compatibility blocker is documented.

**Priority: accurate, lossless conversion of model-visible decision information and target labels, with direct compatibility with the existing R0 retrieval pipeline. Do not invent source fields or silently alter the task semantics.**