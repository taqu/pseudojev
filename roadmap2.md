はい。今回の結果なら **E2 は採用候補としてかなり強い**です。特に hard で accuracy が `0.2388 → 0.3731`、NLL/Brier/ECE/各 margin 指標もすべて改善し、E2+KV なら latency 増も baseline 比 `1.70×` まで抑えられています。report_choice また hard では baseline wrong → E2 correct が11件、逆方向が2件なので、単なる確率の見栄えだけではなく実際の判定も改善しています。report_choice

一方、**`score` に choice と同じ E2 をそのまま適用するのは基本的に意味が薄い、むしろ危険**です。

理由は `score` が nominal classification ではなく **ordinal（順序付き）**だからです。現在の prompt も明示的に、

```text
Possible answers (ordered from lowest to highest):
```

としており、score candidate の順番自体に意味があります。prompt_strategy

さらに `DecisionEngine::expected_level()` も、

```cpp
s += (double)i * probs[i];
```

として、**index 0,1,2,... がそのまま score level 0,1,2,... であることを前提**にしています。decision_engine

なので choice のように、

```text
original:
A = score 0
B = score 1
C = score 2
D = score 3

rotation:
A = score 1
B = score 2
C = score 3
D = score 0
```

とすると、

```text
A: medium
B: high
C: very high
D: low
```

のようになり、**ordinal structure が壊れます**。

モデルに「lowest to highest」と言いつつ cyclic rotation したら完全に矛盾します。

---

ただし、`score` には **別タイプの rotation ensemble** がかなり面白いです。

### semantic order は固定し、candidate token だけ回す

例えば semantic score が

```text
0 = Very poor
1 = Poor
2 = Good
3 = Excellent
```

なら、表示順は絶対に変えない。

Run 1:

```text
A: Very poor
B: Poor
C: Good
D: Excellent
```

Run 2:

```text
B: Very poor
C: Poor
D: Good
A: Excellent
```

Run 3:

```text
C: Very poor
D: Poor
A: Good
B: Excellent
```

Run 4:

```text
D: Very poor
A: Poor
B: Good
C: Excellent
```

重要なのは、画面上・prompt上では常に

```text
Very poor
Poor
Good
Excellent
```

という **semantic ordinal order を維持する**ことです。

変えるのは、

```text
semantic level
↔
candidate token A/B/C/D
```

の binding だけ。

これはかなり意味があります。

目的は choice E2 と同じで、

```text
A が出やすい
B が出にくい
```

という candidate-token prior を semantic score から分離することです。

### 名前も分けた方がいい

僕なら、

```text
E2 = Choice Semantic Rotation Ensemble
```

に対して score は、

```text
Score Label-Rotation Ensemble
```

あるいは

```text
Ordinal Label Ensemble
```

と呼びます。

同じ「rotation」でも性質が違います。

```text
choice:
semantic options themselves rotate through positions

score:
semantic order stays fixed;
only output labels rotate
```

ここは実装でも分けた方が安全です。

---

### 集約は semantic level に戻してから

例えば4レベルなら各 run の logits を、

```text
Run 0
A → level 0
B → level 1
C → level 2
D → level 3

Run 1
B → level 0
C → level 1
D → level 2
A → level 3
```

のように semantic level に戻します。

そして、

```text
S(level_i) =
    mean over runs of logit assigned to level_i
```

としてから softmax。

最後に従来通り、

```text
expected_score =
    Σ i * P(level_i)
```

とすればいいです。

つまり `expected_level()` の考え方自体はそのまま維持できます。

---

### むしろ score では digits の扱いが重要

現在の `NATURAL + score` は、

```cpp
s = std::to_string(i);
```

なので、semantic score 0/1/2/3 と candidate token `"0"/"1"/"2"/"3"` が直結しています。prompt_strategy

これは良い面もあります。

モデルにとって、

```text
0 < 1 < 2 < 3
```

という意味的 prior が使える可能性があります。

逆に悪い面として、

```text
"2" tokenを好む
"3" tokenを嫌う
```

ような token prior と semantic score が完全に混ざります。

なので score では、まず、

```text
digits:
0 1 2 3

letters:
A B C D
```

の比較がかなり重要です。

ロードマップでも score formulation として digits / letters / ordinal labels / semantic anchors を比較する設計になっています。roadmap

その上で、

```text
letters baseline
vs
letters + label rotation ensemble
```

をやると、candidate-token bias の効果をかなりきれいに測れます。

---

僕なら今の研究順序をこう変えます。

```text
noul
  E1 → 保留/モデル依存

choice
  E2 + KV → 採用候補

few-shot
  E3+ → 中止

score
  S0: digits baseline
  S1: letters baseline
  S2: letters + label-rotation ensemble
```

S2では **score descriptions の順序は絶対に回さない**のがポイントです。

そして見るのは accuracy より、

```text
MAE
QWK
NLL
Brier
ECE
expected-score error
ordering consistency
```

です。score は「完全一致率」だけだと、`2` を予測すべき問題で `1` と `0` が同じ1ミスとして扱われてしまうので、MAE/QWK の方がかなり重要です。ロードマップも score 評価に accuracy、MAE、QWK、ordering consistency を置いています。roadmap

なので結論は、

> **choice型の semantic rotation は score にはやらない。**
>
> **ただし ordinal order を保ったまま A/B/C/D のラベルだけ rotation する ensemble は、かなり試す価値がある。**

です。E2 が ternary Bonsai でこれだけ効いた以上、score でも「token binding biasだけを平均化する」実験は次の候補として筋が良いです。