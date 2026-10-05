# Task: Improve `noul` Candidate Binding and Make Letter-Based Selection the Default

We need to improve the `noul` decision path in `pjev`.

The current implementation conflates semantic answers (`Yes` / `No`) with internal candidate tokens. This makes option-order reversal unreliable and can produce contradictory prompts such as:

```text
No: Yes
Yes: No
```

The goal of this change is to separate:

```text
semantic answer
    Å´
candidate label
    Å´
single-token model output
```

The model should preferably choose a neutral one-character candidate label such as `A` or `B`, while the semantic meaning (`Yes` / `No`) remains attached to the candidate separately.

Do not redesign unrelated parts of the system.

## Primary goals

1. Make neutral letter labels (`A`, `B`, `C`, ...) the default candidate representation for `noul`.
2. Preserve the semantic meaning of each `noul` candidate independently from its internal token.
3. Make reversing `Yes` and `No` options actually reverse their semantic binding.
4. Ensure `p_true` is computed from the candidate representing semantic `True`, not from a hard-coded vector index.
5. Keep alternative candidate schemes available for research where practical.
6. Add tests that detect positional/token bias and semantic-binding bugs.

## Current problem

`PromptStrategy::assign_labels()` currently special-cases natural `noul` candidates:

```cpp
if (cfg_.scheme == Scheme::NATURAL && type == "noul") {
    s = (i == 0) ? "No" : "Yes";
}
```

This binds `No` and `Yes` to candidate position instead of semantic meaning.

The prompt builder also derives `No` / `Yes` from the candidate index.

This means reversing the descriptions can create a semantically contradictory prompt while the internal token mapping stays unchanged.

Separately, `DecisionEngine::finish_from_logits()` currently does:

```cpp
if(input.type == "noul")
    out.p_true = probs[1];
```

This assumes candidate index 1 always means semantic `True`.

That assumption must be removed.

## Desired architecture

Treat candidate output labels and semantic meaning as separate concepts.

Conceptually:

```text
Candidate 0
  internal label: "A"
  semantic value: false
  description: "No"

Candidate 1
  internal label: "B"
  semantic value: true
  description: "Yes"
```

If option order is reversed:

```text
Candidate 0
  internal label: "A"
  semantic value: true
  description: "Yes"

Candidate 1
  internal label: "B"
  semantic value: false
  description: "No"
```

The model still outputs only one candidate token, but `p_true` follows the semantic candidate rather than the vector position.

## Candidate representation

Extend `Candidate` so a `noul` candidate can explicitly represent semantic false/true.

Use the project's existing style and types. A possible design is:

```cpp
enum class NoulValue {
    False,
    True
};

struct Candidate {
    std::string key;
    std::string description;
    std::string internal;
    std::optional<NoulValue> noul_value;
};
```

This is only an example. If the existing headers suggest a cleaner representation, use that instead.

Do not infer semantic truth from:

- candidate index,
- candidate label,
- display text,
- token ID.

Store the semantic binding explicitly.

## Prompt strategy

For the default/recommended `noul` path, generate prompts like:

```text
You are performing a classification task.

State:
...

Question:
...

Possible answers:
A: No
B: Yes

Reply with exactly one of A, B and nothing else.
```

If the options are intentionally reversed:

```text
Possible answers:
A: Yes
B: No

Reply with exactly one of A, B and nothing else.
```

The output candidates remain neutral single-character tokens.

Avoid producing prompts where a label and its semantic description conflict, such as:

```text
No: Yes
Yes: No
```

## `assign_labels()`

Refactor `assign_labels()` so label assignment is independent of semantic meaning.

For the normal letter scheme:

```cpp
A
B
C
D
...
```

should be assigned purely by candidate position.

For example:

```cpp
candidates[i].internal =
    std::string(1, static_cast<char>('A' + i));
```

Do not special-case `noul` by automatically assigning internal labels `"No"` and `"Yes"` in the default path.

If `Scheme::NATURAL` is retained as an experimental scheme, make sure its semantics are explicit and do not silently rely on candidate position.

## `p_true`

Remove this assumption:

```cpp
out.p_true = probs[1];
```

Instead, locate the candidate whose semantic `noul` value is `True`.

Conceptually:

```cpp
for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].noul_value == NoulValue::True) {
        out.p_true = probs[i];
        break;
    }
}
```

Adapt the implementation as necessary because `finish_from_logits()` currently receives only keys and candidate IDs.

Prefer passing enough candidate metadata into `finish_from_logits()` rather than reconstructing semantics later.

Do not encode semantic meaning into `keys` as a workaround unless the API contract already defines that clearly.

## Preserve constrained-logit behavior

Do not replace the current constrained-logit mechanism.

The intended decision path remains:

```text
build prompt
Å® evaluate prompt once
Å® extract candidate-token logits
Å® restricted softmax
Å® optional calibration
Å® argmax
```

The model should not be allowed to free-generate arbitrary text.

The change is specifically about cleaner candidate binding and prompt formulation.

## Tests

Add focused tests for `noul`.

At minimum cover the following.

### 1. Normal order

Input semantics:

```text
A: No
B: Yes
```

Verify:

- `A` and `B` are the internal candidate tokens.
- semantic false maps to `A`.
- semantic true maps to `B`.
- `p_true` uses the probability assigned to `B`.

### 2. Reversed order

Input semantics:

```text
A: Yes
B: No
```

Verify:

- semantic true now maps to `A`.
- semantic false maps to `B`.
- `p_true` uses the probability assigned to `A`.

This test is essential.

### 3. Synthetic logits

Use deterministic/mock logits so no real model is required.

Example:

```text
logit(A) = 3.0
logit(B) = 1.0
```

For:

```text
A: No
B: Yes
```

the decision should be semantic false and `p_true` should be the probability of `B`.

For:

```text
A: Yes
B: No
```

the exact same logits should produce semantic true and `p_true` should be the probability of `A`.

This proves semantic interpretation is independent of token position.

### 4. Prompt regression

Assert that generated prompts never contain contradictory mappings such as:

```text
No: Yes
Yes: No
```

for letter-based `noul`.

### 5. Single-token validation

Keep existing candidate-token validation intact.

The selected candidate labels must still tokenize to exactly one token.

## Diagnostic support

Add lightweight debug/test visibility for candidate binding.

It should be possible during testing to inspect:

```text
candidate index
key
description
semantic value
internal label
token ID
raw logit
probability
```

Avoid noisy production logging.

Prefer test-accessible metadata or debug-only logging.

## Benchmark / research behavior

Do not remove the ability to compare candidate schemes.

The roadmap explicitly treats candidate binding and option-order sensitivity as research variables.

The important configurations should remain experimentally comparable, including where practical:

```text
A/B
Yes/No
True/False
other single-token candidates
```

However, the production/default configuration should favor the neutral letter-based scheme unless existing configuration semantics require a compatibility-preserving migration.

## Expected behavior after the change

Given a question whose semantic answer is `Yes`:

```text
A: No
B: Yes
```

the model should ideally prefer:

```text
B
```

If the options are reversed:

```text
A: Yes
B: No
```

the model should ideally prefer:

```text
A
```

The important invariant is:

```text
semantic answer remains Yes
candidate token changes with option order
```

This is how we distinguish actual semantic selection from positional or candidate-token bias.

## Non-goals

Do not:

- change the HTTP API unless required for semantic binding;
- redesign calibration;
- change restricted softmax;
- introduce model replacement;
- optimize KV reuse;
- change the server architecture;
- perform broad refactoring unrelated to candidate semantics.

Keep this change small and testable.

## Deliverables

Implement the change and provide:

1. the code changes;
2. unit tests;
3. a short explanation of the previous bug;
4. the exact new invariant for `noul`;
5. confirmation that normal and reversed Yes/No order both pass;
6. any backward-compatibility impact on `Scheme::NATURAL`;
7. a brief note describing how to benchmark:
   - natural Yes/No candidates,
   - letter candidates,
   - reversed letter candidates.

Before finishing, run the relevant existing tests and any newly added tests.

Prefer the smallest implementation that cleanly separates semantic meaning from candidate-token representation.