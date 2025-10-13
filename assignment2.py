import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = r"C:\Users\grace\OneDrive\桌面\学习资料\SE 4\4AL3\Assignment 2 - Copy\Assignment 2"
RNG = np.random.default_rng(42)

def load_dataset(base_dir, period="2010"):
    sub = "data-2010-15" if period == "2010" else "data-2020-24"
    root = os.path.join(base_dir, sub)

    # 主+时间变化在同一对文件里
    pos_main_tc = np.load(os.path.join(root, "pos_features_main_timechange.npy"))
    neg_main_tc = np.load(os.path.join(root, "neg_features_main_timechange.npy"))
    X_main_tc = np.vstack([pos_main_tc, neg_main_tc])

    # 按列切片：FS-I 是前 18 列；FS-II 是第 19-90 列（索引 18:90）
    X_fsi  = X_main_tc[:, 0:18]
    X_fsii = X_main_tc[:, 18:90]

    # FS-III（历史活动，通常形状 (N,1)）
    pos_hist = np.load(os.path.join(root, "pos_features_historical.npy"))
    neg_hist = np.load(os.path.join(root, "neg_features_historical.npy"))
    X_fsiii = np.vstack([pos_hist, neg_hist])

    # FS-IV（极差）
    pos_mm = np.load(os.path.join(root, "pos_features_maxmin.npy"))
    neg_mm = np.load(os.path.join(root, "neg_features_maxmin.npy"))
    X_fsiv = np.vstack([pos_mm, neg_mm])

    # 标签：正样本=1 在前，负样本=0 在后
    y = np.hstack([
        np.ones(pos_main_tc.shape[0], dtype=int),
        np.zeros(neg_main_tc.shape[0], dtype=int)
    ])

    # 用 data_order 重排，确保与作业的观测顺序一致
    order = np.load(os.path.join(root, "data_order.npy")).astype(int)
    X_fsi  = X_fsi[order]
    X_fsii = X_fsii[order]
    X_fsiii= X_fsiii[order]
    X_fsiv = X_fsiv[order]
    y      = y[order]

    return {"features": {"FS-I": X_fsi, "FS-II": X_fsii, "FS-III": X_fsiii, "FS-IV": X_fsiv},
            "y": y}

def combine_features(features_dict, combo):
    mats = [features_dict[name] for name in combo]
    return np.hstack(mats)

def all_feature_combos(keys=("FS-I","FS-II","FS-III","FS-IV")):
    res = []
    for r in range(1, len(keys)+1):
        for c in combinations(keys, r):
            res.append(c)
    return res

def compute_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return recall - fpr, (tn, fp, fn, tp)

def preprocess_Xy(X, y):
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, scaler

class MySVM:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", n_splits=5, random_state=42):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.n_splits = n_splits
        self.random_state = random_state

    def kfold_tss(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for tr, te in skf.split(X, y):
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]
            # 每折单独标准化（避免数据泄漏）
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
            clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, random_state=self.random_state)
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)
            tss, _ = compute_tss(yte, yhat)
            scores.append(tss)
        scores = np.array(scores, dtype=float)
        return float(scores.mean()), float(scores.std()), scores

def feature_experiment(period="2010"):
    data = load_dataset(BASE, period)
    feats, y = data["features"], data["y"]
    combos = all_feature_combos()

    best = {"combo": None, "mean": -1.0, "std": None}

    for combo in combos:
        X = combine_features(feats, combo)
        Xs, ys, _ = preprocess_Xy(X, y)
        clf = MySVM(C=1.0, kernel="rbf", gamma="scale", n_splits=5, random_state=42)
        mean_tss, std_tss, _ = clf.kfold_tss(Xs, ys)
        print(f"[{period}] {combo}: TSS={mean_tss:.4f} ± {std_tss:.4f}")
        if mean_tss > best["mean"]:
            best = {"combo": combo, "mean": mean_tss, "std": std_tss}
    print(f"\nBest on {period}: {best['combo']} | {best['mean']:.4f} ± {best['std']:.4f}")
    return best


def data_experiment(best_combo):
    for period, title in [("2010", "2010-2015"), ("2020", "2020-2024")]:
        data = load_dataset(BASE, period)
        X = combine_features(data["features"], best_combo)
        Xs, ys, _ = preprocess_Xy(X, data["y"])
        clf = MySVM()
        mean_tss, std_tss, per_fold = clf.kfold_tss(Xs, ys)
        print(f"{title} | {best_combo} : TSS={mean_tss:.4f} ± {std_tss:.4f} | folds={np.round(per_fold,3)}")

best2010 = feature_experiment("2010")
data_experiment(best2010["combo"])
