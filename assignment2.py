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

BASE = r"C:\Users\grace\4AL3-Assignment2"

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
    
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_fold_bars(period, combo, fold_scores, out_dir):
    ensure_dir(out_dir)
    plt.figure()
    plt.bar(np.arange(len(fold_scores)), fold_scores)
    plt.title(f"TSS per fold ({period}) - {'+'.join(combo)}")
    plt.xlabel("Fold")
    plt.ylabel("TSS")
    fname = f"tss_per_fold_{period}_{'_'.join(combo)}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_combo_summary(csv_path, out_dir):
    ensure_dir(out_dir)
    df = pd.read_csv(csv_path)
    df = df.sort_values("mean_tss", ascending=False)
    labels = df["combo"].tolist()
    means  = df["mean_tss"].values
    stds   = df["std_tss"].values

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("TSS (mean ± std)")
    plt.title(f"Feature Set Comparison - {df['period'].iloc[0]}")
    out_path = os.path.join(out_dir, f"combo_summary_{df['period'].iloc[0]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

def one_holdout_confmat(Xs, ys, title, out_path, C=1.0, kernel="rbf", gamma="scale"):
    n = len(ys)
    idx = np.arange(n)
    RNG.shuffle(idx)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xs[tr])
    Xte = scaler.transform(Xs[te])

    clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    clf.fit(Xtr, ys[tr])
    yhat = clf.predict(Xte)
    tss, (tn, fp, fn, tp) = compute_tss(ys[te], yhat)

    disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[tn, fp],[fn, tp]]),
                                  display_labels=[0,1])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title(f"{title}\nTSS={tss:.3f}")
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return tss, (tn, fp, fn, tp)


def feature_experiment(period="2010"):
    out_dir = os.path.join(BASE, f"reports_{period}")
    ensure_dir(out_dir)

    data = load_dataset(BASE, period)
    feats, y = data["features"], data["y"]
    combos = all_feature_combos()

    rows = []
    best = {"combo": None, "mean": -1.0, "std": None}

    for combo in combos:
        X = combine_features(feats, combo)
        Xs, ys, _ = preprocess_Xy(X, y)
        clf = MySVM(C=1.0, kernel="rbf", gamma="scale", n_splits=5, random_state=42)
        mean_tss, std_tss, per_fold = clf.kfold_tss(Xs, ys)
        fig_dir = os.path.join(out_dir, "fold_bars")
        plot_path = plot_fold_bars(period, combo, per_fold, fig_dir)

        rows.append({"period": period, "combo": "+".join(combo),
                     "mean_tss": mean_tss, "std_tss": std_tss})

        print(f"[{period}] {combo}: TSS={mean_tss:.4f} ± {std_tss:.4f}")
        if mean_tss > best["mean"]:
            best = {"combo": combo, "mean": mean_tss, "std": std_tss}
    
    # 保存 CSV（用于报告表格）
    df = pd.DataFrame(rows).sort_values("mean_tss", ascending=False)
    csv_path = os.path.join(out_dir, f"svm_feature_combos_{period}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved summary to: {csv_path}")

    summary_png = plot_combo_summary(csv_path, out_dir)
    print("Saved summary plot:", summary_png)
    
    print(f"\nBest on {period}: {best['combo']} | {best['mean']:.4f} ± {best['std']:.4f}")
    return best


def data_experiment(best_combo, period="2010"):
    # 1) period → 标题
    title = "2010-2015" if period == "2010" else "2020-2024"
    print("="*70)
    print(f"[data_experiment] period={period}  title={title}")
    print("="*70)

    # 2) 加载数据
    data = load_dataset(BASE, period)
    X = combine_features(data["features"], best_combo)
    Xs, ys, _ = preprocess_Xy(X, data["y"])

    # 3) 5 折
    clf = MySVM()
    mean_tss, std_tss, per_fold = clf.kfold_tss(Xs, ys)

    # 4) fold bars（明确使用 period）
    fig_dir = os.path.join(BASE, f"reports_{period}", "fold_bars")
    ensure_dir(fig_dir)
    bars_path = plot_fold_bars(period, best_combo, per_fold, fig_dir)
    print(f"[DEBUG] fold bars will be saved to: {bars_path}")

    # 5) 混淆矩阵（明确使用 period）
    cm_dir = os.path.join(BASE, f"reports_{period}", "confusion_matrices")
    ensure_dir(cm_dir)
    vis_title = f"Confusion Matrix ({title}) - {'+'.join(best_combo)}"
    cm_path = os.path.join(cm_dir, f"cm_{period}_{'_'.join(best_combo)}.png")
    print(f"[DEBUG] confusion matrix will be saved to: {cm_path}")

    tss_holdout, cm_vals = one_holdout_confmat(Xs, ys, vis_title, cm_path)

    # 6) 输出
    print(f"[OK] Saved confusion matrix: {cm_path} | holdout TSS={tss_holdout:.3f}")
    print(f"[RESULT] {title} | {best_combo} : TSS={mean_tss:.4f} ± {std_tss:.4f} | folds={np.round(per_fold,3)}\n")

    # 7) 写 summary.txt（也明确使用 period）
    out_dir = os.path.join(BASE, f"reports_{period}")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {title}\n")
        f.write(f"Feature combination: {'+'.join(best_combo)}\n")
        f.write(f"Mean TSS (5-fold): {mean_tss:.4f} ± {std_tss:.4f}\n")
        f.write(f"Fold scores: {np.round(per_fold,3)}\n")
        f.write(f"Holdout TSS: {tss_holdout:.4f}\n")
        f.write(f"Confusion matrix (tn, fp, fn, tp): {cm_vals}\n")
        f.write(f"Saved figure: {cm_path}\n")
    print(f"[OK] Summary saved to reports_{period}/summary.txt\n")



best2010 = feature_experiment("2010")
best2020 = feature_experiment("2020")
data_experiment(best2010["combo"], "2010")
data_experiment(best2020["combo"], "2020")
