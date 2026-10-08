はい。16個前後を中心に、**「長さ」「種類」「挿入位置」を小さく直交実験**するロードマップが良いです。関連研究でも、フィラーはモデルごとに有効な種類・個数が違い、特に **final `Answer:` の直前**への挿入が有効になりやすいと報告されています。1.7B級でも改善例があります。[arXiv](https://arxiv.org/abs/2509.24884?utm_source=chatgpt.com)

今回の pjev では、300トークンのような長大な filler は避け、**0 / 4 / 8 / 16 / 32** を主軸にするのが良さそうです。

### Phase F0 — Baseline freeze

まず S1 letters を固定します。

```text
score formulation: S1 letters
filler length: 0
answer anchor: current frozen anchor
temperature: T=1 initially
prior correction: off initially
evaluations/item: 1
```

ここを今後の基準にします。

測るものは既存の、

```text
accuracy
MAE
expected-score MAE
QWK
NLL
Brier
ECE
signed margins
large-error rate
latency
prompt tokens
```

です。

---

### Phase F1 — Filler length sweep

最初は filler type を1種類だけに固定します。

第一候補は dots 系です。

```text
. . . . . .
```

または tokenizer を確認した上で、

```text
........
```

です。

長さ：

```text
0
4
8
16
32
```

余裕があれば境界確認用に、

```text
2
64
```

も追加します。

ただし **16を中心値**にします。

目的は、

```text
0 → 8 → 16 で改善
16 → 32 で頭打ち/悪化
```

のような optimum が存在するかを見ることです。

重要なのは「文字数」ではなく **実 token 数を記録すること**です。

例えば `"...."` が tokenizer 上1 tokenなら、16回書いても16 tokensとは限りません。

必ず、

```text
requested filler units
actual filler token count
```

を両方保存します。

---

### Phase F2 — Insertion position

研究上ここはかなり重要です。

同じ16 tokensで、最低3か所比較します。

```text
P0:
... options
[filler]
Answer:

P1:
... options
Answer:
[filler]
[candidate logits]

P2:
... question
[filler]
options
Answer:
```

優先順位は **P0 = Answer: の直前**です。

先行研究でも final `Answer:` の直前が最も有効だったという結果があります。[arXiv](https://arxiv.org/abs/2509.24884?utm_source=chatgpt.com)

pjevなら最終形は例えば、

```text
Possible answers (ordered from lowest to highest):
A: ...
B: ...
C: ...
D: ...

. . . . . . . . . . . . . . . .
Answer:
```

として、その `Answer:` の後で A/B/C/D logits を読む構造です。

これはかなり試す価値があります。

---

### Phase F3 — Filler type

F1/F2で良かった長さ・位置を固定して、種類を比較します。

候補は4種類程度で十分です。

```text
dots:
. . . . . .

ellipsis:
... ... ... ...

neutral counting:
1 2 3 4 5 ...

repeated neutral word:
wait wait wait ...
```

ただし `wait` や `think` のような意味のある単語は prompt semantics を変えるので、僕なら後回しです。

まずは、

```text
dots
repeated punctuation
counting
```

くらい。

研究でも token type によって効き方がかなり異なることが報告されています。[arXiv](https://arxiv.org/abs/2607.22925?utm_source=chatgpt.com)

---

### Phase F4 — 16-token local search

16が良ければ、その近辺だけ細かく見ます。

例えば、

```text
8
12
16
20
24
32
```

です。

ここで最適値を探す。

逆に16で全く改善がなければ、300まで伸ばす必要はないと思います。

このモデルでは「filler computation」自体を使えない可能性もあるからです。

小型モデルで filler が効くケースもありますが、モデル依存性はかなり強いです。[arXiv](https://arxiv.org/abs/2509.24884?utm_source=chatgpt.com)

---

### Phase F5 — Per-item analysis

aggregate metric だけではなく、

```text
S1 correct → filler wrong
S1 wrong → filler correct
```

を必ず出します。

さらに、

```text
expected-score error gain
signed-margin gain
ground-truth probability gain
```

も出します。

理想的なのは、

```text
hard examples improve
easy/original do not regress
```

です。

もし filler が hard だけに効くなら adaptive にできます。

---

### Phase F6 — Adaptive filler

ここはかなり面白いです。

最初に filler なしで1回評価して、

```text
low confidence / low margin
```

なら16 filler版を再評価する。

例えば、

```text
|winner margin| < τ
```

なら filler inference。

構造は、

```text
S1 baseline
   ↓
margin high
   → return

margin low
   ↓
S1 + 16 filler
   ↓
return
```

です。

これなら全件に filler cost を払わなくて済みます。

E2/E1でやった ensemble より軽い可能性があります。

---

### Phase F7 — Calibration

fillerを採用するなら、最後に calibration を別途やり直します。

なぜなら filler によって candidate logits の分布自体が変わるからです。

比較：

```text
S1
S1 + filler16
```

それぞれ別に、

```text
content-free prior
T_score
prior alpha
```

を fitting します。

S1用 calibration を filler版にそのまま流用しない方がいいです。

---

### 実験マトリクス

最初はこれだけで十分です。

| ID | Length | Type | Position |
|---|---:|---|---|
| F0 | 0 | none | — |
| F1 | 4 | dots | before Answer |
| F2 | 8 | dots | before Answer |
| F3 | 16 | dots | before Answer |
| F4 | 32 | dots | before Answer |
| F5 | 16 | dots | after Answer |
| F6 | 16 | counting | before Answer |
| F7 | 16 | alternate punctuation | before Answer |

まずこの8条件。

これで傾向が見えれば十分です。

---

### 採用条件

僕なら filler を採用する条件は厳しめにします。

`score / hard` で、

```text
MAE ↓
expected-score MAE ↓
QWK ↑
NLL/Brier悪化なし
```

のうち少なくとも複数が改善。

originalでは、

```text
accuracy/MAEの明確な regressionなし
```

。

さらに latency 増加が小さいこと。

16 tokens程度なら prompt eval の追加コストは、4-way ensemble よりはるかに小さいはずです。

---

### 重要な判定

この研究で一番知りたいのは、

```text
filler length ↑
→ qualityが一度改善
→ その後plateau / regression
```

という曲線が出るかです。

もし、

```text
0   best
4   worse
8   worse
16  worse
32  worse
```

なら即終了。

逆に、

```text
0   baseline
4   slight +
8   +
16  best
32  plateau
```

ならかなり面白いです。

今回の Bonsai ternary では **まず16 tokens前後まで**という判断はかなり合理的です。大量 filler を入れるより、モデル固有の sweet spot を小さく探索する方が pjev の低レイテンシ設計にも合っています。