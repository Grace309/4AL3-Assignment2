import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ========== CLI PATHS ==========
if len(sys.argv) < 4:
    print("Usage: py -3.12 assignment22.py <path_to_2010_data> <path_to_2020_data> <output_folder>")
    sys.exit(1)

DATA_ROOTS = {"2010": sys.argv[1], "2020": sys.argv[2]}
OUT_BASE = sys.argv[3]
os.makedirs(OUT_BASE, exist_ok=True)

RNG = np.random.default_rng(42)

# ========== FUNCTIONS ==========
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def load_dataset(period="2010"):
    root = DATA_ROOTS[period]
    pos_main_tc = np.load(os.path.join(root, "pos_features_main_timechange.npy"))
    neg_main_tc = np.load(os.path.join(root, "neg_features_main_timechange.npy"))
    X_main_tc = np.vstack([pos_main_tc, neg_main_tc])
    X_fsi, X_fsii = X_main_tc[:, 0:18], X_main_tc[:, 18:90]
    pos_hist = np.load(os.path.join(root, "pos_features_historical.npy"))
    neg_hist = np.load(os.path.join(root, "neg_features_historical.npy"))
    X_fsiii = np.vstack([pos_hist, neg_hist])
    pos_mm = np.load(os.path.join(root, "pos_features_maxmin.npy"))
    neg_mm = np.load(os.path.join(root, "neg_features_maxmin.npy"))
    X_fsiv = np.vstack([pos_mm, neg_mm])
    y = np.hstack([np.ones(pos_main_tc.shape[0]), np.zeros(neg_main_tc.shape[0])])
    order = np.load(os.path.join(root, "data_order.npy")).astype(int)
    return {
        "features": {"FS-I": X_fsi[order], "FS-II": X_fsii[order],
                     "FS-III": X_fsiii[order], "FS-IV": X_fsiv[order]},
        "y": y[order]
    }

def combine_features(f, combo): return np.hstack([f[name] for name in combo])
def all_feature_combos(keys=("FS-I","FS-II","FS-III","FS-IV")):
    from itertools import combinations
    return [c for r in range(1,len(keys)+1) for c in combinations(keys,r)]
def compute_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    recall, fpr = tp/(tp+fn), fp/(fp+tn)
    return recall-fpr, (tn,fp,fn,tp)
def preprocess_Xy(X,y):
    scaler = StandardScaler()
    mask = np.isfinite(X).all(axis=1)
    Xs = scaler.fit_transform(X[mask])
    return Xs, y[mask], scaler

class MySVM:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", n_splits=5):
        self.C, self.kernel, self.gamma, self.n_splits = C, kernel, gamma, n_splits
    def kfold_tss(self,X,y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores=[]
        for tr,te in skf.split(X,y):
            sc=StandardScaler(); Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
            clf=SVC(C=self.C,kernel=self.kernel,gamma=self.gamma,random_state=42)
            clf.fit(Xtr,y[tr]); yhat=clf.predict(Xte)
            tss,_=compute_tss(y[te],yhat); scores.append(tss)
        scores=np.array(scores)
        return float(scores.mean()), float(scores.std()), scores

def plot_fold_bars(period, combo, scores, out_dir):
    ensure_dir(out_dir)
    plt.bar(np.arange(len(scores)), scores)
    plt.title(f"TSS per fold ({period}) - {'+'.join(combo)}")
    plt.xlabel("Fold"); plt.ylabel("TSS")
    path=os.path.join(out_dir,f"tss_per_fold_{period}_{'_'.join(combo)}.png")
    plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close(); return path

def plot_combo_summary(csv_path, out_dir):
    df=pd.read_csv(csv_path).sort_values("mean_tss",ascending=False)
    x=np.arange(len(df)); means, stds=df["mean_tss"],df["std_tss"]
    plt.figure(figsize=(10,5))
    plt.bar(x,means,yerr=stds,capsize=4)
    plt.xticks(x,df["combo"],rotation=60,ha="right")
    plt.ylabel("TSS (mean ± std)")
    plt.title(f"Feature Set Comparison - {df['period'].iloc[0]}")
    out=os.path.join(out_dir,f"combo_summary_{df['period'].iloc[0]}.png")
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close(); return out

def one_holdout_confmat(Xs, ys, title, out_path):
    idx=np.arange(len(ys)); RNG.shuffle(idx); cut=int(0.8*len(ys))
    tr,te=idx[:cut],idx[cut:]
    sc=StandardScaler(); Xtr=sc.fit_transform(Xs[tr]); Xte=sc.transform(Xs[te])
    clf=SVC(C=1.0,kernel="rbf",gamma="scale",random_state=42)
    clf.fit(Xtr,ys[tr]); yhat=clf.predict(Xte)
    tss,cm=compute_tss(ys[te],yhat)
    disp=ConfusionMatrixDisplay(confusion_matrix=np.array([[cm[0],cm[1]],[cm[2],cm[3]]]),display_labels=[0,1])
    fig,ax=plt.subplots(); disp.plot(ax=ax); ax.set_title(f"{title}\nTSS={tss:.3f}")
    ensure_dir(os.path.dirname(out_path)); plt.savefig(out_path,dpi=150,bbox_inches="tight"); plt.close()
    return tss, cm

def feature_experiment(period):
    out_dir = os.path.join(OUT_BASE, f"reports_{period}")
    ensure_dir(out_dir)
    data = load_dataset(period)
    feats, y = data["features"], data["y"]
    rows = []
    best = {"combo": None, "mean": -1}

    for combo in all_feature_combos():
        X = combine_features(feats, combo)
        Xs, ys, _ = preprocess_Xy(X, y)
        clf = MySVM()
        mean, std, scores = clf.kfold_tss(Xs, ys)
        plot_fold_bars(period, combo, scores, os.path.join(out_dir, "fold_bars"))
        rows.append({"period": period, "combo": '+'.join(combo), "mean_tss": mean, "std_tss": std})

        # ✅ 新增：对每个组合都生成 confusion matrix
        cm_dir = os.path.join(out_dir, "confusion_matrices")
        ensure_dir(cm_dir)
        cm_path = os.path.join(cm_dir, f"cm_{period}_{'_'.join(combo)}.png")
        title = f"Confusion Matrix ({period}) - {'+'.join(combo)}"
        tss_holdout, cm = one_holdout_confmat(Xs, ys, title, cm_path)
        print(f"[{period}] Confusion matrix saved for {combo} | holdout TSS={tss_holdout:.3f}")

        # ✅ 写每个组合的 summary
        with open(os.path.join(out_dir, f"summary_{'_'.join(combo)}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Dataset: {period}\nFeature combination: {'+'.join(combo)}\n")
            f.write(f"Mean TSS: {mean:.4f} ± {std:.4f}\nFold scores: {np.round(scores,3)}\n")
            f.write(f"Holdout TSS: {tss_holdout:.4f}\nConfusion matrix: {cm}\n")

        if mean > best["mean"]:
            best = {"combo": combo, "mean": mean, "std": std}

        print(f"[{period}] {combo}: {mean:.4f} ± {std:.4f}")

    df = pd.DataFrame(rows).sort_values("mean_tss", ascending=False)
    csv = os.path.join(out_dir, f"svm_feature_combos_{period}.csv")
    df.to_csv(csv, index=False)
    plot_combo_summary(csv, out_dir)
    print(f"\nBest on {period}: {best['combo']} | {best['mean']:.4f} ± {best['std']:.4f}")
    return best


def data_experiment(combo, period):
    title="2010-2015" if period=="2010" else "2020-2024"
    data=load_dataset(period); X=combine_features(data["features"],combo)
    Xs,ys,_=preprocess_Xy(X,data["y"]); clf=MySVM(); mean,std,scores=clf.kfold_tss(Xs,ys)
    plot_fold_bars(period,combo,scores,os.path.join(OUT_BASE,f"reports_{period}","fold_bars"))
    cm_dir=os.path.join(OUT_BASE,f"reports_{period}","confusion_matrices")
    ensure_dir(cm_dir)
    cm_path=os.path.join(cm_dir,f"cm_{period}_{'_'.join(combo)}.png")
    tss_holdout,cm=one_holdout_confmat(Xs,ys,f"Confusion Matrix ({title}) - {'+'.join(combo)}",cm_path)
    with open(os.path.join(OUT_BASE,f"reports_{period}","summary.txt"),"w",encoding="utf-8") as f:
        f.write(f"Dataset: {title}\nFeature combination: {'+'.join(combo)}\n")
        f.write(f"Mean TSS: {mean:.4f} ± {std:.4f}\nFold scores: {np.round(scores,3)}\n")
        f.write(f"Holdout TSS: {tss_holdout:.4f}\nConfusion matrix: {cm}\n")

# ========== MAIN ==========
best2010=feature_experiment("2010")
best2020=feature_experiment("2020")
data_experiment(best2010["combo"],"2010")
data_experiment(best2020["combo"],"2020")