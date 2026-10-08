import json, math, sys

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def apply_alpha(raw_logits, prior_logits, alpha):
    corr = [r - alpha * p for r, p in zip(raw_logits, prior_logits)]
    return softmax(corr)

def expected_score(probs):
    return sum(i * p for i, p in enumerate(probs))

def qwk(gts, preds, n_cats):
    n = len(gts)
    if n == 0: return 0.0
    O = [[0]*n_cats for _ in range(n_cats)]
    for g, p in zip(gts, preds):
        O[g][p] += 1
    E = [[0.0]*n_cats for _ in range(n_cats)]
    gt_c = [0]*n_cats; pd_c = [0]*n_cats
    for g, p in zip(gts, preds):
        gt_c[g] += 1; pd_c[p] += 1
    for i in range(n_cats):
        for j in range(n_cats):
            E[i][j] = gt_c[i] * pd_c[j] / n
    W = [[(i-j)**2/(n_cats-1)**2 for j in range(n_cats)] for i in range(n_cats)]
    num = sum(W[i][j]*O[i][j] for i in range(n_cats) for j in range(n_cats))
    den = sum(W[i][j]*E[i][j] for i in range(n_cats) for j in range(n_cats))
    return 1.0 - num/den if den > 0 else 0.0

def compute_metrics(items, prior_logits=None, alpha=None):
    n = len(items)
    if n == 0: return {}
    correct = 0; abs_errs = 0.0; exp_abs_errs = 0.0; nll_s = 0.0; brier_s = 0.0
    margins = []; predictions = []; gts = []

    for item in items:
        gt = item['ground_truth']
        n_lv = len(item['levels'])

        if prior_logits is not None and alpha is not None:
            probs = apply_alpha(item['raw_logits'], prior_logits[:n_lv], alpha)
        else:
            probs = item['probs']

        pred = probs.index(max(probs))
        exp_sc = expected_score(probs)
        gts.append(gt); predictions.append(pred)

        ae = abs(pred - gt); abs_errs += ae; exp_abs_errs += abs(exp_sc - gt)
        if ae == 0: correct += 1

        gt_p = probs[gt]
        nll_s += -math.log(max(gt_p, 1e-15))
        brier_s += sum((p - (1.0 if i == gt else 0.0))**2 for i, p in enumerate(probs))

        max_other = max(probs[i] for i in range(len(probs)) if i != pred)
        margins.append(probs[pred] - max_other)

    bins = [[] for _ in range(10)]
    for item in items:
        n_lv2 = len(item['levels'])
        if prior_logits is not None and alpha is not None:
            probs = apply_alpha(item['raw_logits'], prior_logits[:n_lv2], alpha)
        else:
            probs = item['probs']
        pred = probs.index(max(probs))
        conf = max(probs); acc = 1 if pred == item['ground_truth'] else 0
        bins[min(int(conf*10), 9)].append((conf, acc))
    ece = sum(len(b)/n * abs(sum(x[0] for x in b)/len(b) - sum(x[1] for x in b)/len(b)) for b in bins if b)

    sm = sorted(margins)
    n_cats = max(max(len(item['levels']) for item in items), 4)
    large_err = sum(1 for p,g in zip(predictions,gts) if abs(p-g)>=2) / n

    return {
        'accuracy': correct/n, 'mae': abs_errs/n, 'exp_mae': exp_abs_errs/n,
        'qwk': qwk(gts, predictions, n_cats),
        'nll': nll_s/n, 'brier': brier_s/n, 'ece': ece,
        'margin_mean': sum(margins)/n, 'margin_median': sm[n//2],
        'margin_p10': sm[max(0,int(n*0.1)-1)], 'margin_min': min(margins),
        'large_err': large_err,
        'predictions': predictions, 'gts': gts,
    }

for split_name, path in [('original', 'D:/Projects/Go/pseudojev/results/f2/original/score-formulation.json'),
                          ('hard', 'D:/Projects/Go/pseudojev/results/f2/hard/score-formulation.json')]:
    with open(path) as f:
        d = json.load(f)
    runs = {r['experiment']: r for r in d['runs']}

    f2_run = runs['letters-filler-corrected']
    prior_logits = f2_run['score_candidates'][0]['prior_logits']
    f2_items = f2_run['score_items']
    f1_items = runs['letters-filler']['score_items']
    f0_items = runs['letters']['score_items']

    print("\n" + "="*70)
    print(f"SPLIT: {split_name}  (n={len(f2_items)})")
    print("="*70)
    print(f"F1-compatible prior logits: A={prior_logits[0]:.4f} B={prior_logits[1]:.4f} C={prior_logits[2]:.4f} D={prior_logits[3]:.4f}")

    print(f"\n{'Run':<32} {'Acc':>6}  {'MAE':>6}  {'ExpMAE':>8}  {'QWK':>6}  {'NLL':>6}  {'Brier':>6}  {'ECE':>6}")
    for rn in ['natural','letters','letters-rotation','letters-filler','letters-filler-corrected']:
        items = runs[rn]['score_items']
        m = compute_metrics(items)
        label = {'natural':'S0','letters':'S1(F0)','letters-rotation':'S2','letters-filler':'F1','letters-filler-corrected':'F2(a=1)'}.get(rn,rn)
        print(f"  {label:<30} {m['accuracy']:>6.4f}  {m['mae']:>6.4f}  {m['exp_mae']:>8.4f}  {m['qwk']:>6.4f}  {m['nll']:>6.4f}  {m['brier']:>6.4f}  {m['ece']:>6.4f}")

    print(f"\nAlpha sweep (F2 items, {split_name}):")
    print(f"{'alpha':>6}  {'Acc':>6}  {'MAE':>6}  {'ExpMAE':>8}  {'QWK':>6}  {'NLL':>6}  {'Brier':>6}  {'ECE':>6}  {'MarMean':>9}  {'LrgErr':>8}")
    sweep_results = {}
    for alpha in [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
        m = compute_metrics(f2_items, prior_logits, alpha)
        sweep_results[alpha] = m
        print(f"{alpha:>6.2f}  {m['accuracy']:>6.4f}  {m['mae']:>6.4f}  {m['exp_mae']:>8.4f}  {m['qwk']:>6.4f}  {m['nll']:>6.4f}  {m['brier']:>6.4f}  {m['ece']:>6.4f}  {m['margin_mean']:>9.4f}  {m['large_err']:>8.4f}")

    print(f"\nSelection frequencies ({split_name}):")
    n_lv = len(f2_items[0]['levels'])
    print(f"  {'level':>5}  {'F0(S1)':>8}  {'F1':>8}  {'F2(a=1)':>9}  {'GT':>6}")
    for lv in range(n_lv):
        f0_cnt = sum(1 for it in f0_items if it['prediction'] == lv)
        f1_cnt = sum(1 for it in f1_items if it['prediction'] == lv)
        f2_cnt = sum(1 for p in sweep_results[1.00]['predictions'] if p == lv)
        gt_cnt = sum(1 for it in f2_items if it['ground_truth'] == lv)
        print(f"  {lv:>5}  {f0_cnt:>8}  {f1_cnt:>8}  {f2_cnt:>9}  {gt_cnt:>6}")

    f1_correct = [it['correct'] for it in f1_items]
    f2_preds = sweep_results[1.00]['predictions']
    gts = [it['ground_truth'] for it in f2_items]
    f2_correct = [p == g for p, g in zip(f2_preds, gts)]
    f0_correct = [it['correct'] for it in f0_items]

    print(f"\nF1 -> F2(a=1) transitions ({split_name}):")
    print(f"  correct -> wrong: {sum(1 for c1,c2 in zip(f1_correct,f2_correct) if c1 and not c2)}")
    print(f"  wrong -> correct: {sum(1 for c1,c2 in zip(f1_correct,f2_correct) if not c1 and c2)}")
    print(f"  unchanged:        {sum(1 for c1,c2 in zip(f1_correct,f2_correct) if c1==c2)}")

    print(f"\nF0 -> F2(a=1) transitions ({split_name}):")
    print(f"  correct -> wrong: {sum(1 for c1,c2 in zip(f0_correct,f2_correct) if c1 and not c2)}")
    print(f"  wrong -> correct: {sum(1 for c1,c2 in zip(f0_correct,f2_correct) if not c1 and c2)}")
    print(f"  unchanged:        {sum(1 for c1,c2 in zip(f0_correct,f2_correct) if c1==c2)}")

    # Prior flattening diagnostics
    print(f"\nPrior flattening at alpha=1:")
    raw_range = max(prior_logits[:4]) - min(prior_logits[:4])
    corr_logits = [p - 1.0*p for p in prior_logits[:4]]
    corr_range = max(corr_logits) - min(corr_logits)
    raw_var = sum((p - sum(prior_logits[:4])/4)**2 for p in prior_logits[:4]) / 4
    corr_var = sum((c - sum(corr_logits)/4)**2 for c in corr_logits) / 4
    print(f"  raw prior range={raw_range:.4f} variance={raw_var:.4f}")
    print(f"  corrected prior range={corr_range:.4f} variance={corr_var:.4f}")

    # Performance
    eval_ms_list = [it.get('eval_ms', 0) for it in f2_items if it.get('eval_ms', 0) > 0]
    pt_list = [it.get('prompt_tokens', 0) for it in f2_items if it.get('prompt_tokens', 0) > 0]
    ev_list = [it.get('evaluations', 0) for it in f2_items]
    if not eval_ms_list:
        m = f2_run.get('metrics', {})
        print(f"\nMetrics eval_ms not per-item; run metrics: {m.get('eval_ms','?')} prompt_tokens={m.get('prompt_tokens','?')}")
    else:
        print(f"\nPerformance: avg eval_ms={sum(eval_ms_list)/len(eval_ms_list):.1f} avg_prompt_tokens={sum(pt_list)/len(pt_list):.0f} avg_evals={sum(ev_list)/len(ev_list):.2f}")
