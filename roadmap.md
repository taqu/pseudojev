## pjev Revised Roadmap

### Phase 0 — Experimental Decision Core ✅

目的は、Bonsai 1.7B + llama.cpp で constrained-logit 型の Jev decision が成立するか検証すること。

実証済み：

- `noul`
- `choice`
- `score`
- single-token candidate validation
- candidate-logit extraction
- restricted softmax
- deterministic argmax
- constrained one-token generation
- English / Japanese smoke tests
- JevBench baseline
- latency / RSS measurement

結論は、

```text
mechanism: viable
quality: not yet sufficient
```

という状態。元の baseline は choice 0.554、noul 0.649、score 0.444 / MAE 0.78。roadmap

---

### Phase 1 — Minimal Jev-Compatible Native C++ Server

ここから正式実装は **native C++**。

基本構造：

```text
HTTP API
   ↓
Jev compatibility layer
   ↓
DecisionEngine
   ↓
PromptStrategy
   ↓
LlamaBackend
   ↓
llama.cpp
```

主要 endpoint：

```text
POST /api/v1/systemone/
```

重要事項：

- JevBench から Jev / pjev を endpoint 切替だけで比較可能にする
- `DecisionEngine` を HTTP から独立させる
- `PromptStrategy` を独立させる
- `state-first / state-last`
- `natural / letters`
  などを固定しない
- user content と chat special tokens の tokenization を分離
- `<|im_end|>` 等を user content から special token として解釈させない
- CMake を正式 build system とする

Phase 1 の本来の exit criterion も「Jev endpoint と pseudojev endpoint を切り替えて同じ test suite を実行できること」です。roadmap

---

### Phase 2 — Decision Quality Baseline

目的：

> Bonsai 1.7B が JevBench でなぜ失点しているかを、model capability と decision formulation に分解する。

モデル交換はまだ行わない。

研究対象：

**2A — Candidate Binding**

比較：

```text
A/B/C/D
0/1/2/3
Yes/No
True/False
custom single-token IDs
```

**2B — Option Order Sensitivity**

```text
original
reverse
random permutations
```

見るもの：

```text
prediction stability
probability variance
accuracy variance
semantic answer consistency
```

**2C — Prompt Layout**

```text
state → question → options
question → options → state
question → state → options
```

少数の明確な template だけを比較する。

**2D — Candidate Prior Correction**

content-free prompt から candidate prior を測定し、

```text
corrected_logit_i =
    raw_logit_i - prior_logit_i
```

のような補正を検証。

**2E — Score Formulation**

```text
ordinal labels
digits
letters
semantic anchors
```

を比較し、

```text
accuracy
MAE
QWK
ordering consistency
```

で評価。

元ロードマップでも、モデル交換より先に formulation を調べることが明示されています。roadmap

---

### Phase 3 — Calibration + JevBench-Derived Tuning Dataset

ここは当初の calibration フェーズから少し拡張した。

最終目的：

> JevBench dataset から reproducible な pjev tuning dataset を作り、prompt と post-logit parameters の両方を調整できる研究基盤にする。

流れ：

```text
JevBench dataset
        ↓
pjev inference
        ↓
raw candidate logits を保存
        ↓
derived tuning dataset
        ↓
parameter tuning
        ↓
prompt/formulation comparison
        ↓
held-out evaluation
```

derived dataset には最低限：

```text
source ID / split
primitive
state / question / options
ground truth
difficulty
prompt configuration
candidate mapping
candidate token IDs
raw logits
prior logits
corrected logits
probabilities
prediction
```

を保存する。

重要な分離：

**post-logit tuning**

```text
temperature
prior correction strength
```

→ stored logits だけで再評価可能。

**prompt tuning**

```text
layout
candidate scheme
wording
score anchors
```

→ prompt が変わるので llama.cpp inference を再実行。

Calibration はまず：

```text
temperature scaling
```

primitive 別に：

```text
T_noul
T_choice
T_score
```

を許可。

評価：

```text
NLL
Brier
ECE
accuracy
score MAE
```

accuracy と calibration は別物として報告する。これは元ロードマップでも明示されています。roadmap

また tuning / validation / held-out evaluation を明確に分離し、最終評価データへの overfitting を防ぐ。

---

### Phase 4 — Multilingual Validation

正式対象：

```text
English
Japanese
```

目的は翻訳 benchmark そのものではなく、

```text
language degradation
```

を測ること。roadmap

基本手順：

```text
English-selected configuration
        ↓
そのまま Japanese に transfer
        ↓
degradation を測定
        ↓
必要な場合だけ Japanese-specific tuning
```

評価：

- choice / noul / score
- NLL / Brier / ECE
- cross-language semantic consistency
- confidence shift
- `EN correct → JA wrong`
- easy / original / hard
- token count / latency

さらに、

```text
English instruction + Japanese content
Japanese instruction + Japanese content
```

も必要に応じて比較。

英語用 calibration が日本語にも transfer するかも検証する。

---

### Phase 5 — Model Value Gate

ここで初めて、

> Bonsai 1.7B を pjev の default model として採用する価値があるか

を判断する。roadmap

評価軸：

```text
quality
calibration
latency
RSS
model size
multilingual quality
candidate compatibility
```

まず現在の Bonsai を完全に再評価する。

必要な場合のみ、概ね：

```text
0.5B–2B
```

クラスの小型 CPU-friendly GGUF models と比較。

すべて：

```text
pjev
 ↓
llama.cpp
```

の実 product path で比較する。

モデルごとに：

- candidate token compatibility
- model-specific calibration
- English / Japanese degradation
- cold load
- warm latency
- RSS
- GGUF size

を測る。

Phase 5 の結果として **default model decision artifact** を freeze する。

---

### Phase 6 — Production Server

experimental server を実運用可能な local server に hardening。

追加：

```text
graceful shutdown
request validation
timeouts
request size limits
health endpoint
version endpoint
structured errors
basic logging
```

正式起動：

```bash
pjev serve
```

元ロードマップでも「server は引き続き単純にする」とされています。roadmap

方針：

- default bind は localhost
- model は startup 時にロード
- invalid request で crash しない
- user prompt を通常ログに出さない
- inference concurrency は必要なら serialize
- high-concurrency architecture はまだ不要

---

### Phase 7 — Performance Optimization

最大の狙い：

> repeated prompt evaluation を減らす。

元ロードマップでも prompt evaluation がほぼ全コストで、decision math はごく小さいとされています。roadmap

研究対象：

```text
prompt length
KV reuse
multiple-question batching
context sizing
thread count
llama.cpp settings
```

特に：

```text
same state
+
multiple questions
```

を最重要候補とする。

目標：

```text
shared state prefix
   ↓
evaluate once
   ↓
reuse KV
   ↓
question A/B/C...
```

ただし、

- optimized path と reference path を残す
- logits / probabilities / prediction の数値等価性を検証
- state-first が高速でも quality を落とすなら勝手に変更しない
- cross-request persistent KV cache はやらない
- latency と RSS の両方を見る

---

### Phase 8 — Native Distribution

目的：

> development checkout がなくても pjev を配布・実行できる状態にする。

初期完成形：

```text
pjev executable
+
GGUF model
+
versioned config
```

literal な model embedding は必須ではない。

元ロードマップでは Phase 8 で single-binary distribution と model embed / extraction / mmap を扱う想定でした。roadmap

対象：

```text
Linux x86_64
Linux arm64
macOS arm64
Windows x86_64
```

実際に build + smoke test できたものだけ supported とする。

整備：

- release CMake build
- runtime dependency inspection
- artifact naming
- release manifest
- checksums
- model hash
- relocation/path tests
- license/notice files
- archive packaging

基本方針は **adjacent GGUF を先に完成**。

model embedding は、その後に価値がある場合だけ検討。

---

### Phase 9 — Human-Facing CLI

ここで正式に：

```bash
pjev noul ...
pjev choice ...
pjev score ...
```

を作る。

構造：

```text
CLI
 ↓
DecisionEngine
```

HTTP server は経由しない。

元ロードマップでも、CLI と server の decision logic を二重実装しないことが方針です。roadmap

整備：

- human-readable output
- `--json`
- stdout / stderr discipline
- predictable exit codes
- stdin / file input where useful
- Unicode / Japanese support
- CLI ↔ server output equivalence tests

この段階では毎回 model load してよい。

---

### Phase 10 — Background Runtime

目的：

> CLI の repeated model load cost を隠す。

元ロードマップの構造そのまま：roadmap

```text
pjev CLI
 ↓
worker exists?
 ├─ yes → use it
 └─ no  → spawn worker
```

worker：

```text
model loaded
DecisionEngine alive
local IPC
idle timeout
```

idle 後：

```text
unload model
exit
```

追加：

```bash
pjev status
pjev stop
```

IPC：

```text
Unix/macOS → Unix domain socket
Windows    → named pipe
```

重要事項：

- worker は ephemeral helper
- system daemon/service にはしない
- direct CLI path は残す
- local IPC only
- worker は user-scoped
- protocol versioning
- simultaneous startup race 対応
- config/model mismatch を検出
- stale worker recovery
- request content は永続化しない

---

### Phase 11 — UX / GitHub Distribution Polish

最終フェーズ。

元ロードマップの対象：

```text
automatic CPU detection
model extraction/cache
cross-platform packaging
versioning
friendly errors
benchmark command
diagnostics
```

を完成させる。roadmap

今回さらに **GitHub public distribution** を正式スコープに追加。

#### UX

整備：

```bash
pjev --help
pjev --version
pjev diagnostics
pjev diagnostics --json
pjev benchmark
```

friendly errors、CPU/runtime detection、model discovery、worker UX などを最終調整。

#### README.md

public GitHub project の入口として全面整備。

含める：

```text
What is pjev?
Features
Quick Start
GitHub Releases installation
Model setup
noul / choice / score
pjev serve
Background runtime
Build from source
Tests
Configuration
Troubleshooting
Diagnostics
Architecture
Supported platforms
Known limitations
License / model license
```

README のコマンドはすべて実際に検証する。

#### GitHub Actions

最低：

```text
.github/workflows/
  ci.yml
  release.yml
```

必要なら：

```text
real-model.yml
```

通常 CI：

```text
push / PR
 ↓
Linux / macOS / Windows
 ↓
CMake build
 ↓
model-free tests
 ↓
light integration tests
```

巨大 GGUF を毎 PR で落とさない。

Real-model tests は：

```text
release
manual
scheduled
```

などに分離。

Release workflow：

```text
git tag
 ↓
version validation
 ↓
platform builds
 ↓
Phase 8 package
 ↓
smoke tests
 ↓
checksums
 ↓
GitHub Release
```

GitHub Actions 内に別の packaging implementation を作らず、Phase 8 の packaging scripts をそのまま使う。

---

# 全体のクリティカルパス

現在のロードマップを一行でまとめると：

```text
Phase 0
prove mechanism
   ↓
Phase 1
native C++ Jev-compatible foundation
   ↓
Phase 2
understand formulation errors
   ↓
Phase 3
build tuning dataset + calibration
   ↓
Phase 4
validate multilingual transfer
   ↓
Phase 5
freeze default model
   ↓
Phase 6
harden server
   ↓
Phase 7
optimize repeated prompt evaluation
   ↓
Phase 8
make native releases
   ↓
Phase 9
make direct CLI pleasant
   ↓
Phase 10
hide repeated model load
   ↓
Phase 11
polish UX + publish cleanly on GitHub
```

## 一貫して守る設計原則

最終的に重要なのはこのあたりです。

- 実装言語は **native C++**
- command name は **`pjev`**
- llama.cpp を直接使用
- `DecisionEngine` は interface に依存しない
- HTTP / CLI / worker で decision logic を共有
- PromptStrategy / candidate representation を research 可能なまま保つ
- JevBench は単なる最後の benchmark ではなく、Phase 1 以降ずっと research loop の中心
- raw logits を保存し、post-logit tuning は inference をやり直さず行う
- prompt が変わった時だけ inference を再実行
- tuning / validation / holdout を分離
- English → Japanese transfer を multilingual quality の重要指標とする
- model comparison は formulation を十分に詰めた後だけ
- performance optimization は correctness freeze 後
- model distribution と executable distribution を分離
- background worker は daemon ではなく ephemeral local helper
- GitHub Releases を最終的な一般ユーザー向け distribution surface にする
