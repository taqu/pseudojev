はい。**選択肢ローテーションも few-shot も、pjev ではかなり有望**です。ただし、貼ってくれた案には少し強すぎる表現があります。「完全に相殺」「100%指示に従う」「Prefillを完全ゼロ化」は保証できません。実際には、どれだけ効くかを JevBench の held-out で測る、という設計にした方がいいです。

英語専用にする判断自体は、今の研究目的にはかなり合理的です。現在のロードマップでは Phase 4 が English→Japanese transfer の検証になっていますが、英語限定ならそこを製品要件から外し、**英語で formulation/calibration/model quality を最大化することに研究資源を集中**できます。現ロードマップでも candidate representation と option-order sensitivity は Phase 2 の主要研究対象です。roadmap

### 選択肢ローテーションは、かなり期待できる

これは思いつきではなく、LLM の multiple-choice では option ID / position bias が実際に観測されています。ある研究では、選択肢の順序変更で性能が大きく変わり、別の研究では `A/B/C/D` のような option token 自体への prior bias が原因の一つとして特定されています。[arXiv](https://arxiv.org/abs/2309.03882?utm_source=chatgpt.com)

特に `noul` は2択なので実装しやすいです。

1回目：

```text
A: Yes
B: No
Answer:
```

2回目：

```text
A: No
B: Yes
Answer:
```

ただし、単純に

```text
(P_yes_run1 + P_yes_run2) / 2
```

でも動きますが、pjev ならもっと良い方法があります。

**semantic logit margin を合成する**のがおすすめです。

各 run について、

```text
m1 = logit(Yes semantic) - logit(No semantic)
m2 = logit(Yes semantic) - logit(No semantic)
```

と、A/B の位置を semantic Yes/No に戻してから計算します。

そして、

```text
m = (m1 + m2) / 2
P(true) = sigmoid(m / T)
```

とする。

candidate prior correction も使うなら、

```text
m1 =
    (raw_yes_1 - α * prior_yes_1)
  - (raw_no_1  - α * prior_no_1)

m2 =
    (raw_yes_2 - α * prior_yes_2)
  - (raw_no_2  - α * prior_no_2)
```

としてから平均します。

これは確率を後から平均するより、pjev の「logit を直接扱う」という設計に合っています。

### 2択なら「両順序」がかなり安い

`noul` では permutation が2通りしかないので、

```text
A=Yes B=No
A=No  B=Yes
```

を全部評価できます。

choice が4択なら 24 permutation 全部は重いので、そこでは

```text
original
reverse
cyclic rotation × 4
```

くらいから始めるのが現実的です。

例えば4択なら、

```text
run 1: A B C D
run 2: B C D A
run 3: C D A B
run 4: D A B C
```

にすれば、各 semantic option が A/B/C/D を一度ずつ経験します。

これはかなり綺麗です。

各 semantic candidate の log probability を mapping 後に平均して、

```text
score_i =
    mean_r(log P_r(semantic candidate i))
```

最後に softmax する方法が扱いやすいです。

研究でも option permutation を利用した debiasing や voting は有効性が報告されています。[arXiv](https://arxiv.org/abs/2309.03882?utm_source=chatgpt.com)

ただし「完全に bias が消える」とは考えない方がいいです。

---

### Few-shot も効く可能性は十分ある

特に Bonsai 1.7B のような小型モデルでは、

```text
何を判定するのか
どういう形式で答えるのか
A/B が何を意味するのか
```

を例で示すことで、zero-shot より安定する可能性があります。

一方、few-shot はかなり癖があります。例の**選び方・順序・label distribution**によって性能が大きく変わることも知られています。場合によっては example order だけで性能が大幅に変わるという研究もあります。[arXiv](https://arxiv.org/abs/2104.08786?utm_source=chatgpt.com)

だから、

> 3〜5例を入れれば強くなる

と決め打ちしない方がいいです。

pjev ならむしろ、

```text
0-shot
1-shot
2-shot
4-shot
```

を比較するべきです。

そして `noul` なら label balance を守る。

例えば2-shot：

```text
Example 1

State:
The package was delivered successfully.

Question:
Was the package delivered?

Possible answers:
A: No
B: Yes

Answer:
B


Example 2

State:
The server is currently offline.

Question:
Is the server online?

Possible answers:
A: No
B: Yes

Answer:
A


Now answer the following.

State:
{user_state}

Question:
{user_question}

Possible answers:
A: No
B: Yes

Answer:
```

こうすれば、

```text
positive example 1
negative example 1
```

で label frequency を均衡できます。

### さらに一歩進めるなら、few-shot 自体も反転する

ここが pjev では面白いところです。

run 1:

```text
A = No
B = Yes
```

の examples。

run 2:

```text
A = Yes
B = No
```

の examples。

そして query も同じ mapping にする。

つまり、

```text
demonstrations
+
test question
```

全体について candidate binding を反転させます。

これならモデルが

```text
「Bはいつもpositive」
```

と学習するのを防ぎつつ、

```text
example → task semantics
```

だけを利用させられます。

これはかなり試す価値があります。

---

### KV cache のアイデアも良い。ただし「Prefill完全ゼロ」ではない

固定部分、

```text
system/instruction
+
few-shot examples
```

は起動時に評価して KV cache として保持できます。

その後は、

```text
cached fixed prefix
        ↓
user state/question
        ↓
Answer:
```

だけ追加評価できます。

なので、

> 固定 few-shot 部分の repeated prefill を消せる

は正しいです。

ただしユーザー入力そのものは当然 `llama_decode()` が必要なので、**request prefill 全体がゼロになるわけではありません**。

さらに rotation をすると、

```text
cache A:
A=No B=Yes examples

cache B:
A=Yes B=No examples
```

の2本をあらかじめ持てます。

これはかなり面白い構成です。

```text
startup
 ├─ build noul cache AB
 ├─ build noul cache BA
 ├─ build choice rotation cache 0
 ├─ build choice rotation cache 1
 ...
```

としてしまう。

メモリとのトレードオフはありますが、prefix が小さいなら現実的です。

---

### 僕なら JevBench でこの順番に実験します

いきなり全部組み合わせると原因が分からなくなるので、まず baseline を固定します。

```text
English only
letters
no <think>
Answer:
zero-shot
single ordering
```

これを B0 とします。

次に、

```text
B0
↓
+ prior correction
↓
+ temperature calibration
```

ここまでを現在の基準にする。

その後、独立して：

| Experiment | Change |
|---|---|
| E1 | binary order ensemble |
| E2 | cyclic rotation for choice |
| E3 | balanced 2-shot |
| E4 | balanced 4-shot |
| E5 | 2-shot + rotation |
| E6 | 4-shot + rotation |

と比較する。

各々で、

```text
accuracy
NLL
Brier
ECE
latency
prompt tokens evaluated
```

を測る。

accuracy だけ上がって NLL が悪化する場合もありますから、校正品質も残します。

ロードマップでも accuracy と calibration を別々に評価する設計になっています。roadmap

### かなり重要なのは「examples を JevBench test から選ばない」

Few-shot examples まで coding agent に最適化させると、簡単に benchmark leakage が起きます。

なので、

```text
JevBench
   ├─ tuning
   │    └─ few-shot candidatesを選ぶ
   │
   ├─ validation
   │    └─ example setを決定
   │
   └─ held-out
        └─ 最終評価のみ
```

にします。

特に demonstration selection は性能にかなり影響することが報告されているので、few-shot examples 自体を calibration parameter の一種として扱うくらいでちょうどいいです。[arXiv](https://arxiv.org/abs/2104.08786?utm_source=chatgpt.com)

### 英語限定にするなら、設計はむしろかなりシャープになります

最終的に僕なら pjev の方向性を、

```text
English-only input
+
small CPU model
+
neutral one-token labels
+
fixed English prompt
+
optional fixed few-shot prefix
+
permutation/rotation ensemble
+
candidate-prior correction
+
temperature calibration
```

にします。

そしてモデルには「いい文章を生成する能力」を求めない。

求めるのは、

> **候補 token 間の relative logit が、入力 semantics に対して安定して動くこと**

だけです。

これは Bonsai 1.7B のような小型モデルを使う理由ともかなり噛み合います。

そして現状の `blue/dark/noon/midnight` の実験を見る限り、少なくとも **semantic signal は logits に出ている**ので、rotation + calibration + few-shot を試す価値は十分あります。

個人的には次に実装するなら、few-shot より先に **`noul` の2-order logit ensemble** をやります。実装が小さく、bias 除去の効果を一番きれいに測れるからです。その次が balanced 2-shot です。