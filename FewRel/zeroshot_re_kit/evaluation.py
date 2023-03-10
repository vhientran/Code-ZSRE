import numpy as np

def compute_macro_PRF(predicted_idx, gold_idx):
    predicted_idx = np.array(predicted_idx, dtype=int)
    gold_idx = np.array(gold_idx, dtype=int)
    
    i = len(predicted_idx)

    complete_rel_set = set(gold_idx)
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = (predicted_idx[:i] == r)
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
        
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec+avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)
    
    print("Avg_prec, Avg_rec, F1-score on test set: ", avg_prec, avg_rec, f1)
    
    return avg_prec, avg_rec, f1












