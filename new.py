import numpy as np
import skfeature.utility.entropy_estimators as ees

CMI_f_l_l = {}   # (fi, lj, li) -> I(fi; lj | li)
CMI_f_f_l = {}   # (fi, fs, li) -> I(fi; fs | li)

def compute_mi_matrix(data, labels):
    n_samples, n_feats = data.shape
    _, n_labels = labels.shape
    MI = np.zeros((n_feats, n_labels))
    for i in range(n_feats):
        fi = data[:, i]
        for j in range(n_labels):
            MI[i, j] = ees.midd(fi, labels[:, j])
    return MI


def compute_label_weights(labels, eps=1e-12):
    sums = labels.sum(axis=0).astype(float)
    raw = 1.0 / (sums + eps)
    w = raw / raw.sum()
    return w


def compute_ULI_table(MI_matrix, w):
    f_nub = MI_matrix.shape[0]
    ULI = np.zeros(f_nub)

    col_max = MI_matrix.max(axis=0)

    for fi in range(f_nub):
        vec = MI_matrix[fi, :]
        is_self_max = (vec == col_max)

        if is_self_max.any():
            tmp = MI_matrix.copy()
            tmp[fi, :] = -np.inf
            second_max = tmp.max(axis=0)
            max_others = np.where(is_self_max, second_max, col_max)
        else:
            max_others = col_max

        diffs = np.maximum(0.0, vec - max_others)
        ULI[fi] = float(np.dot(w, diffs))

    return ULI


def score_J_fast(f_idx, data, labels, MI_matrix, S_indices, w,
                 CMI_f_l_l, CMI_f_f_l):
    """
    J = A + B - C + ULI
    """

    n_labels = labels.shape[1]

    # ---------- A ----------
    A = float(np.dot(w, MI_matrix[f_idx, :]))

    # ---------- B: label interaction ----------
    B = 0.0
    for li in range(n_labels):
        zi = labels[:, li]
        best = 0.0
        for lj in range(n_labels):
            if li == lj:
                continue
            key = (f_idx, lj, li)
            if key not in CMI_f_l_l:
                CMI_f_l_l[key] = ees.cmidd(
                    data[:, f_idx],
                    labels[:, lj],
                    zi
                )
            I3 = CMI_f_l_l[key]
            if I3 > best:
                best = I3
        B += w[li] * max(0.0, best)

    # ---------- C: redundancy ----------
    C = 0.0
    if S_indices:
        for s in S_indices:
            for li in range(n_labels):
                zi = labels[:, li]
                key = (f_idx, s, li)
                if key not in CMI_f_f_l:
                    CMI_f_f_l[key] = ees.cmidd(
                        data[:, f_idx],
                        data[:, s],
                        zi
                    )
                C += w[li] * CMI_f_f_l[key]
        C /= len(S_indices)

    # ---------- D: ULI ----------
    D = ULI_TABLE[f_idx]

    return A + B - C + D

def LSMFS_BiSearch(data, labels, n_features):
    n_samples, f_nub = data.shape
    _, l_nub = labels.shape

    # ---------- 1. MI ----------
    MI_matrix = compute_mi_matrix(data, labels)
    w = compute_label_weights(labels)

    # ---------- 2. ULI ----------
    global ULI_TABLE
    ULI_TABLE = compute_ULI_table(MI_matrix, w)

    # ---------- 3. 初始相关性 ----------
    rel = np.zeros(f_nub)
    for fi in range(f_nub):
        A = np.dot(w, MI_matrix[fi, :])
        B = 0.0
        for li in range(l_nub):
            zi = labels[:, li]
            best = 0.0
            for lj in range(l_nub):
                if li == lj:
                    continue
                key = (fi, lj, li)
                if key not in CMI_f_l_l:
                    CMI_f_l_l[key] = ees.cmidd(
                        data[:, fi],
                        labels[:, lj],
                        zi
                    )
                best = max(best, CMI_f_l_l[key])
            B += w[li] * max(0.0, best)
        rel[fi] = A + B + ULI_TABLE[fi]

    # ---------- 4. first feature ----------
    F = [int(np.argmax(rel))]

    # ---------- 5. forward selection ----------
    while len(F) < n_features:
        best_score = -1e30
        best_f = None

        for fi in range(f_nub):
            if fi in F:
                continue
            score = score_J_fast(
                fi, data, labels,
                MI_matrix, F, w,
                CMI_f_l_l, CMI_f_f_l
            )
            if score > best_score:
                best_score = score
                best_f = fi

        F.append(best_f)

    return F


if __name__ == "__main__":
    X = np.array([[1, 0, 0, 1, 0, 1, 2, 3],
                  [1, 0, 2, 0, 1, 1, 1, 2],
                  [2, 1, 1, 0, 0, 2, 2, 1],
                  [1, 0, 0, 1, 1, 1, 2, 1],
                  [3, 2, 1, 1, 1, 1, 3, 3],
                  [1, 1, 1, 1, 1, 2, 2, 3],
                  [1, 0, 0, 1, 1, 2, 2, 2],
                  [1, 1, 0, 1, 0, 2, 2, 0],
                  [1, 1, 0, 0, 0, 0, 2, 2],
                  [0, 1, 0, 1, 1, 2, 2, 2]])

    Y = np.array([[1, 0, 0, 1, 1],
                  [1, 1, 0, 1, 0],
                  [0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1],
                  [0, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1],
                  [1, 1, 0, 1, 0]])

    select_nub = 5
    aa = LSMFS_BiSearch(X, Y, select_nub)
    print(aa)