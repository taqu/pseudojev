できます。むしろ **S1 + local embedding + adaptive retrieval few-shot** は、ここまで試した filler より筋がいいと思います。

ポイントは、単なる「毎回4-shot」ではなく、

> **必要なときだけ、意味的に近い例を1〜2件だけ入れる**

ことです。

英語限定なら embedding 側もかなり単純化できます。

### 基本構造

```text
                    ┌─ high confidence ─→ S1そのまま返す
input
  ↓
S1 zero-shot
  ↓
confidence / margin
  ↓
                    └─ uncertain
                          ↓
                    local embedding
                          ↓
                    exemplar retrieval
                          ↓
                    sufficiently similar?
                     /             \
                   no               yes
                   ↓                 ↓
                 S1結果          1〜2 shot追加
                                      ↓
                                   再推論
```

これなら few-shot のコストを全リクエストに払わずに済みます。

---

## なぜ static few-shot より期待できるか

以前中止した few-shot は、固定例を毎回入れる形だと、

```text
example-order bias
label-frequency bias
irrelevant demonstration
prompt length increase
```

が常時乗ります。

Adaptive retrieval なら、

```text
temporal question
→ temporal系の近い例

policy question
→ policy系の近い例

numeric score
→ numeric scoreの近い例
```

という使い方ができます。

小型モデルの場合、一般的な「推論能力をfew-shotで増やす」より、

> **この入力は過去のどの判定パターンに近いかを明示する**

使い方の方が期待できます。

これは pjev の用途とも相性がいいです。

---

# ただし最重要なのは exemplar bank

JevBench全体をそのまま検索対象にしてはいけません。

最低でも、

```text
training / exemplar bank
validation
held-out evaluation
```

を完全分離します。

held-out の類似例や同一テンプレート派生例が exemplar bank に入っていると、かなり簡単に benchmark leakage になります。

特に JevBench に、

```text
same scenario
same source
minor wording variants
paired examples
```

があるなら、source family 単位で分離した方がいいです。

---

# scoreなら検索空間をかなり絞れる

まず `score` 専用で試すのが良いです。

検索対象は最低でも、

```text
primitive == score
number_of_levels == current number_of_levels
language == English
```

に限定。

4-level score に5-level scoreの例を混ぜるのは避けます。

さらに可能なら、

```text
task family
domain
difficulty
```

もmetadataとして持たせます。

---

# 何をembeddingするか

僕なら最初はこれです。

```text
State:
{state}

Question:
{question}

Score meanings:
0: {description}
1: {description}
2: {description}
3: {description}
```

ただし実際の embedding text では、A/B/C/D を入れなくてもいいです。

retrieval の目的は candidate token matching ではなく、**問題意味の類似性**だからです。

例えば：

```text
STATE
The payment was received three days late.

QUESTION
How timely was the payment?

LEVELS
very late
late
on time
early
```

のような canonical retrieval representation を別途作る。

LLM prompt と embedding input は分離して構いません。

---

# 最初は cosine Top-K だけで十分

数十MiB規模なら、最初から複雑な vector DB は不要です。

exemplar 数が数千程度なら、

```text
precomputed float embeddings
+
brute-force cosine similarity
```

でも十分現実的です。

C++なら全 vector を連続メモリに置いて、

```text
dot(query, exemplar_i)
```

するだけ。

必要なら後で HNSW 等にできます。

まず retrieval quality を確認すべきです。

---

# ただし単純 Top-2 には罠がある

例えば上位2件が、

```text
nearest #1 → answer D
nearest #2 → answer D
```

だと、few-shot がD priorを作る可能性があります。

S1/S2で既に candidate label identity がかなり効くことが分かったので、ここは特に注意したいです。

したがって diagnostics として、

```text
retrieved label distribution
```

を必ず保存します。

将来的には、

```text
top semantic match
+
best contrasting example
```

の2-shotも試せます。

---

# 最初は1-shotからやるのがいい

僕ならいきなり4-shotにしません。

実験順は、

```text
R0
S1 zero-shot

R1
nearest 1-shot

R2
nearest 2-shot

R3
adaptive 1-shot

R4
adaptive 1-or-2-shot
```

です。

特に1.7B級なら、**1-shotが一番面白い**と思います。

4件入れて reasoning context を豊富にするより、

```text
「これはこの既知ケースに近い」
```

という強い一例を置く方が、小型モデルでは扱いやすい可能性があります。

---

# Prompt例

S1を維持して：

```text
Here is an example of the same scoring task.

Example:
State:
...

Question:
...

Possible answers (ordered from lowest to highest):
A: ...
B: ...
C: ...
D: ...

Answer: C

Now answer the new case.

State:
...

Question:
...

Possible answers (ordered from lowest to highest):
A: ...
B: ...
C: ...
D: ...

Answer:
```

そして最後の A/B/C/D logits だけを取る。

重要なのは、demonstration側でも **S1と同じ letter mapping** を使うことです。

---

# Adaptive gate は2段階にしたい

単に「marginが低ければ few-shot」だけではなく、

```text
1. S1 uncertainty
2. retrieval similarity
```

の両方を見るのがいいです。

例えば概念的に：

```text
if baseline_margin >= τ_margin:
    return zero_shot

retrieve nearest exemplar

if cosine_similarity < τ_similarity:
    return zero_shot

return few_shot_reinfer()
```

つまり、

> **困っていて、かつ役に立ちそうな例が存在するときだけfew-shot**

です。

この方がかなり安全です。

---

# さらに良いgate

S1では probability calibration が完全ではないので、

```text
max probability
```

だけより、

```text
top1_logit - top2_logit
```

を見る方がいいと思います。

scoreなら、

```text
winner margin
```

または、

```text
entropy
```

との併用でもいいです。

最初は単純に：

```text
top1 - top2 < τ
```

で十分。

τは tuning split だけで決めます。

---

# embedding similarity 自体も有力な特徴になる

面白いのは、

```text
similarity = 0.93
```

のケースと、

```text
similarity = 0.52
```

のケースでは few-shot を同じように使う必要がない点です。

例えば最終的に、

```text
low margin + high similarity
    → use retrieved demonstration

low margin + low similarity
    → stay zero-shot
```

にできます。

後者で適当な例を入れると、むしろ小型モデルを引っ張る可能性があります。

---

# さらに「近い例の答え」を直接利用しないこと

ここは重要です。

KNNとして、

```text
nearest examplesのground truthをvote
```

してしまうと、もはや LLM few-shot の研究なのか retrieval classifier の研究なのか分からなくなります。

最初は明確に、

```text
embedding:
example selection only

Bonsai:
final decision
```

と分けるべきです。

ただし後から比較 baseline として、

```text
embedding kNN classifier
```

を測るのは非常に価値があります。

もし kNN 単体が Bonsai より強かったら、それも重要な結果です。

---

# 実験ロードマップならこうします

### AR0 — Retrieval sanity

推論には使わない。

JevBench training/exemplar bank をembeddingして、各validation queryについて、

```text
top-1 similarity
top-5 similarities
retrieved source IDs
retrieved ground-truth levels
```

だけ出す。

人間が見て、

> 本当に意味的に近い例が取れているか

を確認。

ここが悪ければ few-shot に進まない。

### AR1 — Always 1-shot

最も近いexampleを必ず1件入れる。

```text
S1 zero-shot
vs
S1 + retrieved 1-shot
```

を比較。

adaptive gate はまだ入れない。

これで retrieval demonstration にそもそも価値があるかを見る。

### AR2 — Similarity-gated 1-shot

```text
similarity >= τ_sim
```

のときだけ1-shot。

### AR3 — Confidence + similarity adaptive

```text
baseline uncertain
AND
retrieval similarity high
```

だけ再推論。

これが本命です。

### AR4 — Adaptive 1 vs 2 shot

top-1が非常に近い：

```text
1-shot
```

少し曖昧：

```text
2-shot
```

など。

ただしここまで来るのはAR3に成果が出てからでいいです。

---

# 必須metrics

既存の score metrics に加えて、

```text
retrieval hit rate
mean top-1 similarity
few-shot activation rate
accuracy when activated
accuracy when not activated

zero-shot correct → few-shot wrong
zero-shot wrong → few-shot correct

mean extra prompt tokens
mean model evaluations/request
latency multiplier
```

を追加します。

特に大事なのは、

```text
rescue count
harm count
```

です。

例えば：

```text
baseline wrong → adaptive correct: 12
baseline correct → adaptive wrong: 3
```

ならかなり有望です。

---

# latency面も悪くない

embedding modelが数十MiBでローカルなら、

```text
zero-shot S1
+
必要時だけ embedding retrieval
+
必要時だけ second LLM pass
```

にできます。

ただ、僕ならさらに一歩進めて、**embeddingは最初に常時実行してもよい**と考えます。

小型embeddingのforwardが Bonsai 1.7Bより十分安いなら、

```text
embedding(query)
↓
nearest example + similarity
↓
S1 inference
↓
gate
```

でもいい。

そして再推論が必要なときだけ few-shot。

計測して決めればいいです。

---

## 一番期待している形

最終的にはこうです。

```text
English request
      │
      ├─ local embedding
      │      ↓
      │   nearest exemplar
      │   similarity
      │
      ↓
S1 zero-shot Bonsai
      ↓
semantic margin

margin strong
    → return

margin weak + similarity weak
    → return zero-shot

margin weak + similarity strong
    → retrieved 1-shot rerun
         ↓
       return
```

これはかなり pjev らしい設計です。

**常に重くするのではなく、不確実な判定だけ追加計算を投入する。**

そして今回の filler や S2 と違い、追加 token が意味のある task-specific information なので、計算コストを使う理由もはっきりしています。

僕なら次は filler 系を止めて、この **AR0 → AR1** を試します。AR0で retrieval quality を確認してからAR1に行けば、かなり低リスクで検証できます。