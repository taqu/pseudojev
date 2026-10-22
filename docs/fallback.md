## Additional Requirement — Zero-Shot Fallback for Missing Exemplars

Implement a deterministic zero-shot fallback when the selected task-specific inference strategy cannot obtain the required number of valid exemplars.

### 1. Fallback Policy

The production routing policy remains:

| Task kind | Primary strategy | Required exemplars | Fallback |
|---|---|---:|---|
| `choice` | Nearest 2-shot, Order A | 2 | S1 Zero-shot |
| `noul` | Nearest 1-shot | 1 | S1 Zero-shot |
| `score` | Weighted kNN-5 | 5 | S1 Zero-shot |

If fewer than the required number of eligible exemplars are available, execute the existing S1 zero-shot inference path.

Do not silently reduce 2-shot to 1-shot or kNN-5 to kNN-3/kNN-1.

Do not select arbitrary or ineligible exemplars to fill missing positions.

### 2. Exemplar Eligibility

Count only exemplars that satisfy all existing eligibility requirements:

- Present in the permitted exemplar database and training allowlist.
- Valid one-hot ground-truth answer.
- Matching task kind.
- Matching score-level count, where applicable.
- Not the query itself.
- Not duplicated within the selected exemplars.
- Valid semantic answer mapping to the target query.
- Successfully retrieved and decoded.

An exemplar that fails validation must not count toward the required number.

### 3. Fallback Execution

Implement the following behaviour:

```cpp
Decision decide(const Task& task) {
    const auto strategy = route_by_task_kind(task.kind);
    const auto required = required_exemplars(strategy);

    const auto exemplars = retrieve_valid_exemplars(
        task, required
    );

    if (exemplars.size() < required) {
        return decide_zero_shot(task);
    }

    return execute_strategy(task, strategy, exemplars);
}
```

This is conceptual pseudocode; adapt it to the existing production interfaces.

The zero-shot fallback must reuse the original S1 implementation, including its prompt construction, candidate mapping, and candidate-logit scoring.

Do not implement a separate fallback-specific prompt or scoring method.

### 4. Failure Scope

Trigger zero-shot fallback when:

- The eligible exemplar pool is empty.
- Retrieval returns fewer valid exemplars than required.
- Retrieved exemplars fail eligibility checks.
- Exemplar records cannot be decoded or mapped to the query.

Do not treat every runtime failure as a missing-exemplar condition.

Model-loading failures, invalid input, database corruption, and unrelated inference errors must retain their existing error-handling behaviour.

If the fallback itself fails, propagate the error through the existing production error contract.

### 5. Observability

Record the selected strategy and whether fallback occurred.

Use lightweight structured diagnostics such as:

- Requested strategy
- Available valid exemplar count
- Required exemplar count
- Fallback reason
- Executed strategy

The externally visible decision output must remain compatible with the existing API.

Do not expose experimental metadata in normal user-facing output.

Avoid excessive logging or unnecessary database writes.

### 6. Regression Tests

Add tests verifying:

1. Choice with 2 valid exemplars uses Nearest 2-shot.
2. Choice with 0 or 1 valid exemplar uses S1 Zero-shot.
3. Noul with 1 valid exemplar uses Nearest 1-shot.
4. Noul with 0 valid exemplars uses S1 Zero-shot.
5. Score with 5 valid exemplars uses weighted kNN-5.
6. Score with 0–4 valid exemplars uses S1 Zero-shot.
7. Invalid or duplicated exemplars do not satisfy the required count.
8. Fallback predictions match the existing S1 implementation.
9. Normal-path predictions remain unchanged.
10. Unrelated runtime errors are not silently converted into fallback decisions.

### 7. Acceptance Criteria

- Missing or insufficient exemplars never cause an arbitrary prediction.
- All three task kinds fall back to S1 Zero-shot when required exemplar counts are not met.
- No partial few-shot or reduced-k kNN inference is performed.
- Zero-shot fallback uses the existing validated S1 inference path.
- Normal R4 routing behaviour remains unchanged when sufficient exemplars are available.
- Fallback behaviour is deterministic, tested, and documented.

This fallback is a reliability mechanism, not a new accuracy optimization. Do not conduct additional threshold tuning or change the selected R4 routing policy.