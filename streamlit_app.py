import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error,
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.inspection import permutation_importance


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="BBO Owl Migration App", layout="wide")
st.title("🦉 BBO Owl Migration App (EDA + FE + Modeling + XAI)")


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("1) Upload data")

upload_mode = st.sidebar.radio(
    "Upload method",
    ["Private Manual Upload", "Local/Drive Path"]
)

short_thr = st.sidebar.slider("Short stay threshold (days)", 0.5, 10.0, 3.0, 0.5)
long_thr  = st.sidebar.slider("Long stay threshold (days)", 2.0, 30.0, 7.0, 0.5)

st.sidebar.caption("Use your same thresholds as notebook (default ~3 and 7).")

det_file = None
old_meta_file = None
combined_csv = None

if upload_mode == "Private Manual Upload":
    det_file = st.sidebar.file_uploader("NEW detections (SawWhets...xlsx)", type=["xlsx"])
    old_meta_file = st.sidebar.file_uploader("OLD metadata (clean_df_selected.csv)", type=["csv"])
    combined_csv = st.sidebar.file_uploader("OPTIONAL: combined_sawwhet_owls.csv", type=["csv"])
else:
    det_path = st.sidebar.text_input("Detections xlsx path", "")
    old_path = st.sidebar.text_input("Old metadata csv path", "")
    comb_path = st.sidebar.text_input("Optional combined csv path", "")
    det_file = det_path if det_path else None
    old_meta_file = old_path if old_path else None
    combined_csv = comb_path if comb_path else None


# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_all_sheets_xlsx(xlsx_or_path):
    xls = pd.ExcelFile(xlsx_or_path)
    sheets = xls.sheet_names
    frames = []
    for s in sheets:
        try:
            temp = pd.read_excel(xls, sheet_name=s)
            frames.append(temp)
        except:
            pass
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_csv(csv_or_path):
    return pd.read_csv(csv_or_path, low_memory=False)

def build_datetime(df):
    df = df.copy()
    # standardize col names
    df.columns = [c.strip() for c in df.columns]

    if "DATETIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        return df

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    if "TIME" in df.columns:
        # sometimes TIME already string, sometimes datetime.time
        df["TIME"] = df["TIME"].astype(str)

    if "DATE" in df.columns and "TIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            errors="coerce"
        )
    elif "tsCorrected" in df.columns:
        df["DATETIME"] = pd.to_datetime(df["tsCorrected"], unit="s", errors="coerce")

    return df

def add_hour(df):
    df = df.copy()
    if "DATETIME" in df.columns:
        df["hour"] = df["DATETIME"].dt.hour
    return df

def compute_true_stay(df):
    first_det = df.groupby("motusTagID")["DATETIME"].min()
    last_det  = df.groupby("motusTagID")["DATETIME"].max()
    stay_true = (last_det - first_det).dt.total_seconds() / (3600*24)
    stay_true = stay_true.clip(lower=0).reset_index()
    stay_true.columns = ["motusTagID","stay_duration_days"]
    return stay_true

def aggregate_owl_features(df):
    def safe_mean(s): return pd.to_numeric(s, errors="coerce").mean()
    def safe_std(s):  return pd.to_numeric(s, errors="coerce").std()

    num_cols = ["snr","sigsd","freq","freqsd","slop","burstSlop","antBearing","port","nodeNum","runLen","hour"]
    num_cols = [c for c in num_cols if c in df.columns]

    agg_dict = {"detections_count": ("motusTagID","size")}
    for c in num_cols:
        agg_dict[f"{c}_mean"] = (c, safe_mean)
        agg_dict[f"{c}_std"]  = (c, safe_std)

    owl_features = df.groupby("motusTagID").agg(**agg_dict).reset_index()
    return owl_features

def make_residency_labels(owl_df, short_thr, long_thr):
    bins = [0, short_thr, long_thr, np.inf]
    labels = ["Vagrant","Migrant","Resident"]
    owl_df["ResidencyType_true"] = pd.cut(
        owl_df["stay_duration_days"],
        bins=bins, labels=labels, include_lowest=True
    )
    return owl_df

def safe_split(X, y, task="reg"):
    n = len(X)
    if n < 5:
        st.error(f"Not enough owls to split/train (rows={n}). Adjust thresholds/filters.")
        st.stop()

    stratify_arg = None
    if task == "cls":
        counts = y.value_counts()
        if counts.shape[0] >= 2 and counts.min() >= 2:
            stratify_arg = y
        else:
            st.warning("Too few samples per class → splitting without stratify.")
            stratify_arg = None

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)


# ---------------------------
# Load data
# ---------------------------
if old_meta_file is None:
    st.info("Upload OLD metadata (clean_df_selected.csv) to continue.")
    st.stop()

with st.spinner("Loading old metadata..."):
    df_old = load_csv(old_meta_file)

meta_one = df_old.groupby("motusTagID").first().reset_index()

# If combined CSV exists -> use it directly like notebook
if combined_csv is not None:
    with st.spinner("Loading combined CSV..."):
        df = load_csv(combined_csv)
else:
    if det_file is None:
        st.info("Upload NEW detections xlsx (multi-sheet) or combined CSV.")
        st.stop()

    with st.spinner("Loading NEW detections (all sheets) ..."):
        df = load_all_sheets_xlsx(det_file)

if df.empty:
    st.error("Detections file loaded empty. Check the file.")
    st.stop()

df = build_datetime(df)
df = add_hour(df)

# ---------------------------
# Build modeling dataset (same as notebook)
# ---------------------------
stay_true = compute_true_stay(df)
owl_features = aggregate_owl_features(df)

owl_df = owl_features.merge(stay_true, on="motusTagID", how="left")
owl_df = owl_df.merge(meta_one, on="motusTagID", how="left", suffixes=("","_meta"))

owl_df = owl_df.dropna(subset=["stay_duration_days"]).copy()
owl_df = make_residency_labels(owl_df, short_thr, long_thr)


st.success(" Data pipeline finished automatically!")


# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["📌 Overview", "📊 EDA", "🧠 Modeling", "🧩 XAI"])


# ============ OVERVIEW ============
with tabs[0]:
    st.subheader("Shapes")
    c1,c2,c3 = st.columns(3)
    c1.metric("Detections rows", df.shape[0])
    c2.metric("Unique owls", df["motusTagID"].nunique())
    c3.metric("Final owl_df rows", owl_df.shape[0])

    st.subheader("Final dataset preview")
    st.dataframe(owl_df.head(20))

    st.subheader("Stay duration distribution")
    fig, ax = plt.subplots()
    ax.hist(owl_df["stay_duration_days"], bins=30)
    ax.set_xlabel("Stay duration (days)")
    ax.set_ylabel("Count")
    st.pyplot(fig)


# ============ EDA ============
with tabs[1]:
    st.subheader("Residency counts (True)")
    st.bar_chart(owl_df["ResidencyType_true"].value_counts())

    st.subheader("Numeric summary")
    st.write(owl_df.describe())

    num_cols = owl_df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 2:
        st.subheader("Correlation heatmap (numeric)")
        corr = owl_df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10,6))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=90)
        ax.set_yticks(range(len(num_cols)))
        ax.set_yticklabels(num_cols)
        fig.colorbar(im)
        st.pyplot(fig)


# ============ MODELING ============
with tabs[2]:
    st.subheader("Regression: Predict stay_duration_days")

    # Prepare regression features exactly like notebook
    y_reg = owl_df["stay_duration_days"]
    drop_cols = ["motusTagID","stay_duration_days","ResidencyType_true"]
    X_reg = owl_df.drop(columns=[c for c in drop_cols if c in owl_df.columns], errors="ignore")

    # keep numeric only
    X_reg = X_reg.select_dtypes(include=np.number)
    X_reg = X_reg.dropna(axis=1, how="all")

    # median impute
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_reg), columns=X_reg.columns)

    X_train, X_test, y_train, y_test = safe_split(X_imp, y_reg, task="reg")

    reg_models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=2.0, epsilon=0.2)),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    }

    rows = []
    pred_store = {}

    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        rows.append({"Model": name, "RMSE": rmse, "R2": r2})
        pred_store[name] = pred

    reg_results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    st.dataframe(reg_results)

    best_reg_name = reg_results.iloc[0]["Model"]
    best_reg = reg_models[best_reg_name]

    st.write(f" Best Regression Model: **{best_reg_name}**")

    # fit on full data + predict
    best_reg.fit(X_imp, y_reg)
    owl_df["predicted_stay_days"] = best_reg.predict(X_imp)

    # scatter plot true vs predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred_store[best_reg_name])
    ax.set_xlabel("True stay duration (days)")
    ax.set_ylabel("Predicted stay duration (days)")
    st.pyplot(fig)


    st.markdown("---")
    st.subheader("Classification: Predict ResidencyType_true")

    # X for classification (numeric only, same style)
    X_cls = X_imp.copy()
    y_cls = owl_df["ResidencyType_true"].astype(str)

    # label encode
    le = LabelEncoder()
    y_cls_enc = le.fit_transform(y_cls)

    Xc_train, Xc_test, yc_train, yc_test = safe_split(X_cls, y_cls_enc, task="cls")

    candidates = {
        "LogReg": {
            "estimator": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
            ]),
            "params": {
                "clf__C": np.logspace(-2,2,12),
                "clf__solver": ["lbfgs","liblinear"]
            }
        },
        "SVC": {
            "estimator": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", SVC(class_weight="balanced"))
            ]),
            "params": {
                "clf__C": np.logspace(-2,2,8),
                "clf__kernel": ["rbf","poly"]
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
            "params": {
                "n_estimators": [200,400,600],
                "max_depth": [4,6,8,None],
                "min_samples_split": [2,5,10]
            }
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [200,400],
                "learning_rate": [0.01,0.05,0.1],
                "max_depth": [2,3,4]
            }
        }
    }

    search_results = []
    best_models = {}

    for name, cfg in candidates.items():
        with st.spinner(f"Tuning {name}..."):
            search = RandomizedSearchCV(
                cfg["estimator"],
                cfg["params"],
                n_iter=15,
                cv=3,
                scoring="f1_weighted",
                random_state=42,
                n_jobs=-1
            )
            search.fit(Xc_train, yc_train)
            best_models[name] = search.best_estimator_

            ypred = search.best_estimator_.predict(Xc_test)
            acc = accuracy_score(yc_test, ypred)
            f1 = f1_score(yc_test, ypred, average="weighted")

            search_results.append({
                "Model": name,
                "BestParams": search.best_params_,
                "Test_Accuracy": acc,
                "Test_F1": f1
            })

    cls_results = pd.DataFrame(search_results).sort_values("Test_F1", ascending=False)
    st.dataframe(cls_results)

    best_cls_name = cls_results.iloc[0]["Model"]
    best_cls = best_models[best_cls_name]

    st.write(f" Best Classification Model: **{best_cls_name}**")

    yc_pred = best_cls.predict(Xc_test)

    st.text("Classification Report:")
    st.text(classification_report(yc_test, yc_pred, target_names=le.classes_))

    cm = confusion_matrix(yc_test, yc_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_)
    ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

    # store for XAI tab
    st.session_state["best_reg"] = best_reg
    st.session_state["best_cls"] = best_cls
    st.session_state["X_imp"] = X_imp
    st.session_state["y_reg"] = y_reg
    st.session_state["Xc_test"] = Xc_test
    st.session_state["yc_test"] = yc_test
    st.session_state["le"] = le


# ============ XAI ============
with tabs[3]:
    st.subheader("Explainability (XAI)")

    if "best_reg" not in st.session_state:
        st.info("Train models first in Modeling tab.")
        st.stop()

    best_reg = st.session_state["best_reg"]
    best_cls = st.session_state["best_cls"]
    X_imp = st.session_state["X_imp"]
    y_reg = st.session_state["y_reg"]
    Xc_test = st.session_state["Xc_test"]
    yc_test = st.session_state["yc_test"]
    le = st.session_state["le"]

    st.markdown("### Regression Feature Importance (Permutation)")
    try:
        r = permutation_importance(best_reg, X_imp, y_reg, n_repeats=10, random_state=42)
        imp_vals = pd.Series(r.importances_mean, index=X_imp.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        imp_vals.head(15).plot(kind="bar", ax=ax)
        ax.set_ylabel("Importance")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Regression permutation importance failed: {e}")

    st.markdown("### Classification Feature Importance (Permutation)")
    try:
        r2 = permutation_importance(best_cls, Xc_test, yc_test, n_repeats=10, random_state=42)
        imp_vals2 = pd.Series(r2.importances_mean, index=Xc_test.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        imp_vals2.head(15).plot(kind="bar", ax=ax)
        ax.set_ylabel("Importance")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Classification permutation importance failed: {e}")

    st.markdown("### SHAP (optional)")
    try:
        import shap

        # Regression SHAP (sampled for speed)
        sample_X = X_imp.sample(min(200, len(X_imp)), random_state=42)
        explainer = shap.Explainer(best_reg, sample_X)
        shap_values = explainer(sample_X)

        fig = plt.figure()
        shap.summary_plot(shap_values, sample_X, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.info("SHAP not installed / not supported on cloud — acceptable for submission.")
        st.caption(str(e))
