# F1 Filler Tokenization Diagnostics

Model: `bonsai.gguf`

## Candidates tested

| Filler string                              | chars | tokens | tokens/char |
|--------------------------------------------|-------|--------|-------------|
| `................` (16 periods)             |  16   |    1   |  0.06       |
| `. . . . . . . .` (8 period-space pairs)   |  15   |    8   |  0.53       |
| `... ... ... ...` (4 ellipsis-space)        |  15   |    4   |  0.27       |
| `. . . . . . . . . . . . . . . .` (16 ×)   |  31   |   16   |  0.52       |

## Frozen F1 filler

```
filler_text:         ". . . . . . . . . . . . . . . ."
filler_char_count:   31
filler_token_count:  16
insertion_position:  end of user content, immediately before ANSWER_ANCHOR
```

Each `. ` (period + space) tokenizes as one token in the Bonsai tokenizer.  
The 16-period run merges to a single token (BPE merging).  
The ellipsis-space `... ` merges to one token.

Method: measured as `score_items[*].prompt_tokens` delta on `jevbench/datasets/public/hard.jsonl`
with `--scheme letters --calibration ""`.
