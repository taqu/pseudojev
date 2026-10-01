JevBench は `choice / noul / score` を分けて評価でき、`overall / choice / noul` は exact accuracy、`score` は QWK と MAE を使っています。現在の公開ゴールドサンプルは232件で、Jevは overall 88.8%、choice 94%、noul 92%、score-QWK 0.92 / MAE 0.20 と報告されています。[GitHub](https://github.com/model-collapse/jev-bench?utm_source=chatgpt.com)  
またJev本体のAPIは現在 `POST /api/v1/systemone/`、1リクエスト最大20 questions、`noul / choice / score` の3型です。[JEV AI](https://jev-ai.org/docs/?utm_source=chatgpt.com)

なので、私はこんな順番を勧めます。

# pseudojev Roadmap

## Phase 0 — Experimental Core

**目的:** Bonsai 1.7B + llama.cpp で「Jev型decision」が成立するかを最小コードで確認する。

この段階ではGoもHTTPも不要です。既にlogits取得と1-token constrained decodingが確認できているので、残るコアは3 primitiveです。

```text
state + question
       ↓
prompt builder
       ↓
Bonsai 1.7B
       ↓
restricted candidate logits
       ↓
softmax
       ↓
DecisionResult
```

実装対象は、

```text
noul
  → 2 candidates

choice
  → N candidates

score
  → K ordered candidates
  → probability distribution
  → expected value
```

まで。

内部candidateはユーザー文字列ではなく、

```text
A B C D ...
```

あるいは確実にsingle-tokenになるIDを使います。

### Exit criteria

JevBenchの小さなsubsetを流して、

- 3種類すべて動く
- NaN等が出ない
- probability sum ≈ 1
- deterministicに再現可能
- Bonsaiが少なくとも「ランダムより十分上」

を確認。

ここでは精度を追い込みません。

---

# Phase 1 — Minimal Jev-Compatible Server

**ここから正式なpseudojev MVP。**

目的は、

> **JevBenchからJevとpseudojevを同じように呼べる**

状態を作ることです。

Go executableとして、

```bash
pseudojev serve
```

だけ実装。

```text
HTTP
 ↓
Jev request parser
 ↓
decision engine
 ↓
Bonsai / llama.cpp
 ↓
Jev-compatible response
```

### API

まずは公式APIに合わせて、

```text
POST /api/v1/systemone/
```

を実装します。現在のJevでは `state` と最大20個のquestionsを受けます。[JEV AI](https://jev-ai.org/docs/?utm_source=chatgpt.com)

サポート対象：

```text
noul
choice
score
```

この段階では認証やbilling互換は不要です。

むしろ、

```text
Authorization
usage billing
idempotency
rate limit
```

などは意図的に後回し。

### response compatibility

最低限、

```json
{
  "model": "pseudojev-bonsai-1.7b",
  "answers": {
    ...
  }
}
```

として、

```text
noul:
  noul

choice:
  choice
  probabilities
  confidence

score:
  score
  legend
  probabilities
  confidence
```

をJevと互換にします。

scoreはJev同様、

```text
Σ(index × probability)
```

の**decimal expected value**にする。これはJevの現行仕様とも一致します。[JEV AI](https://jev-ai.org/docs/?utm_source=chatgpt.com)

### Exit criteria

JevBench adapterから、

```text
Jev
pseudojev
```

を交換可能にする。

---

# Phase 2 — Benchmark Baseline

ここが最初の大きなGo/No-Goポイントです。

JevBench全体を、

```text
Jev
vs
pseudojev / Bonsai 1.7B
```

で測定。

最低でも、

| Metric | Jev | pseudojev |
|---|---:|---:|
| overall accuracy | | |
| choice accuracy | | |
| noul accuracy | | |
| score QWK | | |
| score MAE | | |
| p50 latency | | |
| p95 latency | | |
| RSS | | |

を残します。

加えてpseudojev独自で、

```text
cold load time
model load RSS
warm request latency
questions/sec
```

も測る。

### この段階ではcalibrationを分離して見る

特に、

```text
accuracy
```

と

```text
probability quality
```

を混ぜないことが重要です。

まず、

> 「モデルは判断自体をできているか」

を見る。

その後、

> 「そのlogitを確率として信用できるか」

を見る。

---

# Phase 3 — Inference Optimization

Phase 2で「可能性あり」と判断できた場合のみ進みます。

ここではモデル変更より先に、**decision formulationを詰める**のが良いです。

例えば、

```text
prompt format
candidate token
criteria formatting
question ordering
state formatting
```

を実験。

特に、

```text
A/B/C/D
0/1/2/3
専用token
```

によって結果が変わる可能性があります。

既存のローカルJev互換実装でも、scoreのverbalizerによって性能差がかなり出ており、bare digitが `"Level i"` やlabel textを上回った実験が公開されています。[GitHub](https://github.com/us/jev-local?utm_source=chatgpt.com)

これはpseudojevでもかなり重要になるはずです。

### このPhaseで見るもの

```text
candidate order sensitivity
option count sensitivity
prompt sensitivity
state length sensitivity
language sensitivity
```

です。

そして、

```text
English
Japanese
Chinese
Spanish
...
```

についてJevBenchを翻訳したsubsetを用意する。

ここで初めて**「多言語CPU decision model」**という狙いを検証します。

---

# Phase 4 — Calibration

accuracyが許容範囲に来てから実施。

```text
raw logits
 ↓
softmax
```

だけで十分か確認します。

不十分なら、

```text
temperature scaling
```

を第一候補にする。

つまり、

```text
softmax(logits / T)
```

です。

primitiveごとに、

```text
T_noul
T_choice
T_score
```

を持ってもいい。

ただしモデルやtaskごとに大量のmagic parameterを持ち始めたら、それは危険信号です。

### Benchmark

ここでは、

```text
NLL
Brier score
ECE
```

を追加。

特にpseudojevは、

> probabilityを返す製品

なので、単純accuracyだけでは足りません。

---

# Phase 5 — Value Gate

ここで一度、開発を止めて判断します。

私はここをかなり明確なgateにします。

例えば暫定的に、

```text
Jev accuracyの90～95%以上
+
Jevより十分安い/ローカル
+
CPUで実用的latency
+
多言語で大崩れしない
```

ならGO。

重要なのは、

**「Jevと完全同等でなければ失敗」ではない**

ことです。

pseudojevには、

```text
offline
private
free
CPU only
no API key
single binary
```

という別軸の価値があります。

例えば、

```text
Jev      89%
pseudojev 84%
```

でも、

```text
local
offline
250MB model
CPU
```

なら十分面白い可能性があります。

逆に、

```text
overall 55%
```

ならCLIやdaemonを作る前に、モデル/decision formulationを再考すべきです。

---

# Phase 6 — Production-quality Server

価値が確認できたら、初めてserverを製品品質にします。

追加するものは、

```text
graceful shutdown
request limits
timeouts
concurrency control
structured logging
health endpoint
version endpoint
```

あたり。

例えば、

```text
GET /health
GET /version
POST /api/v1/systemone/
```

だけで十分。

ここでもまだdaemonにはしません。

実行は、

```bash
pseudojev serve
```

。

これでserver用途だけなら完成品として使える状態にします。

---

# Phase 7 — Single Binary Distribution

ここで、

```text
Go
+
llama.cpp static link
+
Bonsai GGUF
```

を1配布物にします。

重要なのは、

**Phase 1からsingle binary化を完成させようとしないこと**

だと思います。

最初は、

```text
pseudojev
bonsai.gguf
```

でも構わない。

JevBenchで勝負できることが分かってから、

```text
pseudojev
```

だけにする。

これなら、埋め込みmodel extraction / mmap / platform buildといった周辺課題が研究フェーズを邪魔しません。

---

# Phase 8 — CLI

ここでようやくCLIです。

例えば、

```bash
pseudojev noul \
  --state "..." \
  --question "Is this urgent?"
```

```bash
pseudojev choice \
  --state ticket.txt \
  --question "Route this ticket" \
  --choice billing="..." \
  --choice technical="..."
```

```bash
pseudojev score \
  --state review.txt \
  --question "Rate sentiment" \
  --level "Very negative" \
  --level "Negative" \
  --level "Neutral" \
  --level "Positive"
```

出力はデフォルトJSONでいいと思います。

CLI内部はserverと別実装にせず、

```text
CLI frontend
     │
     ▼
same DecisionEngine
     ▲
     │
HTTP frontend
```

にする。

---

# Phase 9 — Persistent Background Process

ここで初めてロード時間を隠蔽します。

CLI実行時、

```text
pseudojev
  ↓
local daemon exists?
  ├─ yes → request
  └─ no
      ↓
     spawn
      ↓
     model load
      ↓
     request
```

通信は、

```text
Unix domain socket
```

Windowsなら、

```text
named pipe
```

など。

daemon自身も同じバイナリ：

```bash
pseudojev internal-daemon
```

あるいはhidden subcommand。

### lifecycle

```text
start
 ↓
model load
 ↓
requests
 ↓
idle timer
 ↓
10 min no requests
 ↓
exit
```

ここで、

```bash
pseudojev status
pseudojev stop
```

を追加。

---

# Phase 10 — UX Polish

最後です。

この段階で初めて、

```bash
curl .../pseudojev
chmod +x pseudojev
pseudojev ...
```

だけで動く世界を完成させます。

候補として、

```text
automatic daemon
automatic model extraction
automatic CPU feature detection
AVX2 / AVX512 selection
Metal support
```

など。

---

## 全体を縮めると

私は大きく4段階に分けます。

```text
STAGE A — Can it work?

Phase 0  Decision core
Phase 1  Minimal Jev server
Phase 2  JevBench baseline


STAGE B — Is it good?

Phase 3  Inference optimization
Phase 4  Calibration
Phase 5  Value Gate


STAGE C — Can we ship it?

Phase 6  Production server
Phase 7  Single binary


STAGE D — Can we make it delightful?

Phase 8  CLI
Phase 9  Background process
Phase 10 UX polish
```

そして**一番重要なのはPhase 5で、本当に止まって判断すること**だと思います。

今回のプロジェクトの場合、Go CLIやdaemonを書くこと自体はそれほど大きな技術リスクではありません。本当のリスクは、

```text
Bonsai 1.7B
+
constrained logits
```

が、

```text
Jev-like decision quality
```

にどこまで近づけるかです。

なので最初のマイルストーンは「pseudojev CLI完成」ではなく、

> **Bonsai 1.7BでJevBenchを走らせ、Jevとの差を数字で出す**

に置くのがいいです。

その結果が良ければ、その後の **Go + static llama.cpp + embedded Bonsai + invisible daemon** は、かなり明確な製品化ロードマップになります。[GitHub](https://github.com/model-collapse/jev-bench?utm_source=chatgpt.com)
