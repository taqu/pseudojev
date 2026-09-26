# Task: Add English Comments to Go Source Files

Review all `*.go` files in the project and add clear, concise English comments to improve code readability and maintainability.

## Goals

Add useful comments for:

* Structs and other important type definitions.
* Functions and methods.
* Important fields when their purpose is not immediately obvious.
* Significant logical blocks inside functions.

Comments should explain the **purpose, intent, or reasoning** behind the code rather than simply restating the code itself.

## Scope

Process all `*.go` files in the repository, except for:

* Generated files.
* Vendored dependencies.
* Third-party source code.
* Files or directories that are clearly not intended to be manually maintained.

Do not modify generated code.

## Struct and Type Comments

Add an English comment to exported structs, interfaces, aliases, and other significant type definitions.

Follow normal Go documentation conventions where appropriate.

Example:

```go
// Server manages incoming client connections and coordinates request handling.
type Server struct {
    listener net.Listener
    workers  *WorkerPool
}
```

For important unexported types, also add comments when the role of the type is not obvious from its name.

Avoid comments that merely repeat the declaration.

Bad:

```go
// Server is a server.
type Server struct {
```

Better:

```go
// Server coordinates network connections and dispatches requests to workers.
type Server struct {
```

## Function and Method Comments

Add a concise English comment to functions and methods that describes:

* What the function does.
* Important inputs or assumptions, when relevant.
* Important side effects.
* Significant return-value behavior or error conditions, when they are not obvious.

For exported functions and methods, follow Go documentation conventions and begin the comment with the identifier when practical.

Example:

```go
// LoadConfig reads and validates the application configuration from the given path.
func LoadConfig(path string) (*Config, error) {
```

For small private helper functions whose behavior is already completely obvious, avoid adding unnecessary documentation purely for completeness.

## Comments Inside Functions

Add comments at the **logical block level**, not at the individual-line level.

A comment should normally describe a meaningful group of statements such as:

* Input validation.
* State initialization.
* Data collection or preprocessing.
* Main processing loops.
* Filtering or transformation.
* Error recovery.
* Resource acquisition or cleanup.
* Synchronization.
* Cache handling.
* Result construction.
* Fallback behavior.

Example:

```go
func processRequest(req *Request) error {
    // Validate the request before accessing dependent resources.
    if err := validateRequest(req); err != nil {
        return err
    }

    // Load all records required by the processing stage.
    records, err := loadRecords(req.ID)
    if err != nil {
        return err
    }

    // Process valid records and accumulate the final result.
    for _, record := range records {
        if !record.Valid {
            continue
        }

        processRecord(record)
    }

    return nil
}
```

Do **not** comment every statement.

Avoid comments like:

```go
// Increment i.
i++

// Check if err is not nil.
if err != nil {
    return err
}
```

Instead, describe the surrounding intent when useful.

## Comment Style

Use concise, natural English.

Prefer comments that answer:

* Why is this block necessary?
* What is this section trying to accomplish?
* What assumption does this code depend on?
* Why is this implementation structured this way?
* What non-obvious behavior should a future maintainer know?

Avoid:

* Translating individual statements into English.
* Explaining basic Go syntax.
* Excessively long comments.
* Speculation about behavior that cannot be confirmed from the code.
* Comments that may quickly become inconsistent with the implementation.

If the purpose of a piece of code cannot be determined confidently, do not invent an explanation.

## Preserve Existing Behavior

This task is primarily a documentation change.

Do not intentionally change:

* Program behavior.
* APIs.
* Control flow.
* Data structures.
* Error-handling semantics.
* Concurrency behavior.

Do not refactor unrelated code just to make commenting easier.

Small formatting changes caused by `gofmt` are acceptable.

## Existing Comments

Preserve useful existing comments.

You may:

* Correct inaccurate or outdated English comments.
* Improve unclear comments.
* Consolidate redundant comments.
* Replace low-value comments with more meaningful block-level explanations.

Do not remove important information merely to make the commenting style uniform.

## Comment Density

Aim for moderate comment density.

As a general rule:

* Important structs/types: comment them.
* Important functions/methods: comment them.
* Complex functions: add comments before major logical blocks.
* Simple functions: usually require only a function-level comment, if any.
* Straightforward code should remain mostly self-explanatory.

The result should **not** look like every few lines were mechanically annotated.

## Workflow

For each Go source file:

1. Understand the file's purpose before adding comments.
2. Identify important types, structs, interfaces, functions, and methods.
3. Add or improve documentation comments where useful.
4. Inspect non-trivial functions and identify their major logical blocks.
5. Add concise comments before those blocks.
6. Remove or rewrite comments that merely repeat the code.
7. Run `gofmt` on modified Go files.
8. Ensure the changes do not alter program behavior.
9. Run the project's existing Go tests or relevant validation commands when available.

## Final Verification

Before completing the task, verify that:

* Comments are written in English.
* Important structs and functions are documented.
* Complex function bodies have block-level comments where appropriate.
* Comments explain intent rather than syntax.
* There are no unnecessary line-by-line comments.
* No generated or third-party code was modified.
* The code still passes `gofmt`.
* Existing tests continue to pass.
* No unrelated behavioral or structural changes were introduced.
