import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

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

warnings.filterwarnings("ignore")

# =========================
# Page config
# =========================
st.set_page_config(page_title="BBO Owl Migration App", layout="wide")
st.title("🦉 BBO Owl Migration App (EDA + FE + Modeling + XAI)")


# =========================
# Sidebar controls
# =========================
st.sidebar.header("1) Upload data")

upload_mode = st.sidebar.radio(
    "Upload method",
    ["Private Manual Upload", "Local/Drive Path"]
)

st.sidebar.caption(
    "Residency thresholds are fixed to 0–3 days (Vagrant), "
    "3–7 days (Migrant), 7+ days (Resident), same as the notebook."
)

combined_csv = None
old_meta_file = None

if upload_mode == "Private Manual Upload":
    combined_csv = st.sidebar.file_uploader(
        "FINAL combined dataset (combined_sawwhet_owls.csv)",
        type=["csv"]
    )
    old_meta_file = st.sidebar.file_uploader(
        "OLD metadata (clean_df_selected.csv)",
        type=["csv"]
    )
else:
    comb_path = st.sidebar.text_input("combined_sawwhet_owls.csv path", "")
    old_path = st.sidebar.text_input("clean_df_selected.csv path", "")
    combined_csv = comb_path if comb_path else None
    old_meta_file = old_path if old_path else None


# =========================
# Helper functions
# =========================
@st.cache_data(show_spinner=False)
def load_csv(csv_or_path):
    return pd.read_csv(csv_or_path, low_memory=False)


def safe_split(X, y, task="reg"):
    """Train-test split with safety for tiny datasets / imbalanced classes."""
    n = len(X)
    if n < 5:
        st.error(f"Not enough owls to split/train (rows={n}). Adjust thresholds/filters.")
        st.stop()

    stratify_arg = None
    if task == "cls":
        counts = pd.Series(y).value_counts()
        if counts.shape[0] >= 2 and counts.min() >= 2:
            stratify_arg = y
        else:
            st.warning("Too few samples per class → splitting without stratify.")
            stratify_arg = None

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)


# =========================
# Load datasets
# =========================
if combined_csv is None or old_meta_file is None:
    st.info("Upload BOTH combined_sawwhet_owls.csv and clean_df_selected.csv to continue.")
    st.stop()

with st.spinner("Loading combined dataset..."):
    df = load_csv(combined_csv)

with st.spinner("Loading old metadata..."):
    df_old = load_csv(old_meta_file)

# ============================================================
# STEP 3 – Identify and Standardize the Motus Tag ID Column
# ============================================================
df = df.copy()
df.columns = [c.strip() for c in df.columns]

possible_id_cols = [c for c in df.columns if "motus" in c.lower() and "tag" in c.lower()]
possible_id_cols += [c for c in df.columns if c.lower() in ["tag_id", "tagid", "motustagid_sheet"]]

if not possible_id_cols:
    st.error(
        "Could not find a Motus Tag ID column in combined_sawwhet_owls.csv.\n\n"
        f"Columns found: {list(df.columns)}"
    )
    st.stop()

id_col = possible_id_cols[0]
df.rename(columns={id_col: "motusTagID"}, inplace=True)
df["motusTagID"] = pd.to_numeric(df["motusTagID"], errors="coerce")

st.write("🔎 Possible ID cols in NEW detections:", possible_id_cols)
st.write("Unique owls in NEW detections:", df["motusTagID"].nunique())

# ============================================================
# STEP 3 – Cleaning and Building a Valid DATETIME Column
# ============================================================
if "DATETIME" in df.columns:
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
else:
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    else:
        df["DATE"] = pd.NaT

    if "TIME" in df.columns:
        df["TIME_clean"] = df["TIME"].astype(str).str.extract(r"(\d{1,2}:\d{2}:\d{2})")[0]
        df["TIME_clean"] = pd.to_timedelta(df["TIME_clean"], errors="coerce")
    else:
        df["TIME_clean"] = pd.to_timedelta(np.nan)

    df["DATETIME"] = df["DATE"] + df["TIME_clean"]

# hour + date_only for visualizations
df["hour"] = df["DATETIME"].dt.hour
df["date_only"] = df["DATETIME"].dt.date

valid_dt = df["DATETIME"].notna().sum()
st.write("Valid DATETIME rows:", valid_dt)

# ============================================================
# STEP 4 – Compute True Stay Duration From RAW Detection Timestamps
# ============================================================
first_det = df.groupby("motusTagID")["DATETIME"].min()
last_det = df.groupby("motusTagID")["DATETIME"].max()

stay_true = (last_det - first_det).dt.total_seconds() / (3600 * 24)
stay_true = stay_true.clip(lower=0)
stay_true = stay_true.reset_index()
stay_true.columns = ["motusTagID", "stay_duration_days"]

st.write("True stay durations computed for owls:", stay_true.shape[0])
st.dataframe(stay_true.head())

# ============================================================
# STEP 5 – Aggregating Owl-Level Features
# ============================================================
def safe_mean(s):
    return pd.to_numeric(s, errors="coerce").mean()


def safe_std(s):
    return pd.to_numeric(s, errors="coerce").std()


num_cols = [
    "snr", "sigsd", "freq", "freqsd", "slop", "burstSlop",
    "antBearing", "port", "nodeNum", "runLen", "hour"
]
num_cols = [c for c in num_cols if c in df.columns]

agg_dict = {"detections_count": ("motusTagID", "size")}
for c in num_cols:
    agg_dict[f"{c}_mean"] = (c, safe_mean)
    agg_dict[f"{c}_std"] = (c, safe_std)

owl_features = df.groupby("motusTagID").agg(**agg_dict).reset_index()

st.write("Owl-level feature table shape:", owl_features.shape)
st.dataframe(owl_features.head())

# ============================================================
# STEP 6 – Load Old Metadata & Standardize motusTagID
# ============================================================
df_old = df_old.copy()
df_old.columns = [c.strip() for c in df_old.columns]

possible_old_ids = [c for c in df_old.columns if "motus" in c.lower() and "tag" in c.lower()]
possible_old_ids += [c for c in df_old.columns if c.lower() in ["tag_id", "tagid", "motustagid_sheet"]]

if not possible_old_ids:
    st.error(
        "Could not find a Motus Tag ID column in clean_df_selected.csv.\n\n"
        f"Columns found: {list(df_old.columns)}"
    )
    st.stop()

old_id_col = possible_old_ids[0]
df_old.rename(columns={old_id_col: "motusTagID"}, inplace=True)
df_old["motusTagID"] = pd.to_numeric(df_old["motusTagID"], errors="coerce")

meta_one = df_old.groupby("motusTagID").first().reset_index()

st.write("Old metadata shape:", df_old.shape)
st.write("Old metadata unique owls:", meta_one["motusTagID"].nunique())
st.dataframe(meta_one.head())

# ============================================================
# STEP 7 – Merging Features, True Stay Duration & Metadata
# ============================================================
owl_df = owl_features.merge(stay_true, on="motusTagID", how="left")
owl_df = owl_df.merge(meta_one, on="motusTagID", how="left", suffixes=("", "_meta"))

st.write("Final dataset shape:", owl_df.shape)
st.write("Final unique owls:", owl_df["motusTagID"].nunique())
st.dataframe(owl_df.head())

# ============================================================
# STEP 8 – Categorizing Owls into Residency Types (fixed thresholds)
# ============================================================
# Use fixed thresholds exactly like notebook:
# 0–3 days  -> Vagrant
# 3–7 days  -> Migrant
# 7+ days   -> Resident
short_thr = 3     # short stay threshold in days
long_thr = 7      # long stay threshold in days

bins = [0, short_thr, long_thr, np.inf]
labels = ["Vagrant", "Migrant", "Resident"]

owl_df["ResidencyType_true"] = pd.cut(
    owl_df["stay_duration_days"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

st.write("ResidencyType_true counts:")
st.write(owl_df["ResidencyType_true"].value_counts())

st.success("Data pipeline (Steps 2–8) completed as in the notebook!")

# =========================
# Tabs
# =========================
tabs = st.tabs(["📌 Overview", "📊 EDA", "🧠 Modeling", "🧩 XAI"])

# =========================
# OVERVIEW TAB
# =========================
with tabs[0]:
    st.subheader("Shapes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Combined rows", df.shape[0])
    c2.metric("Unique owls (raw)", df["motusTagID"].nunique())
    c3.metric("Final owl_df rows", owl_df.shape[0])

    st.subheader("Final dataset preview")
    st.dataframe(owl_df.head(20))

    st.subheader("Stay duration distribution (days)")
    fig, ax = plt.subplots()
    ax.hist(owl_df["stay_duration_days"], bins=30)
    ax.set_xlabel("Stay duration (days)")
    ax.set_ylabel("Owls")
    st.pyplot(fig)

# =========================
# EDA TAB
# =========================
with tabs[1]:
    st.subheader("True Residency Type Distribution (New Motus Data)")
    st.bar_chart(owl_df["ResidencyType_true"].value_counts())

    # Top 15 owls by number of days stayed
    st.subheader("Top 15 Owls by Number of Days Stayed")
    top15 = owl_df.nlargest(15, "stay_duration_days")[["motusTagID", "stay_duration_days"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(top15["motusTagID"].astype(str), top15["stay_duration_days"])
    ax.set_xlabel("Owl ID")
    ax.set_ylabel("Stay Duration (days)")
    ax.set_title("Top 15 Owls by Number of Days Stayed")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # What time of day owls are most active
    st.subheader("What Time of Day Owls Are Most Active")
    if "hour" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(df["hour"].dropna(), bins=24)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Detections")
        ax.set_title("What Time of Day Owls Are Most Active")
        st.pyplot(fig)

    # Daily owl activity at the station (line plot)
    st.subheader("Daily Owl Activity at the Station")
    if "date_only" in df.columns:
        daily = df.groupby("date_only")["motusTagID"].count().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily["date_only"], daily["motusTagID"], marker="o")
        ax.set_title("Daily Owl Activity at the Station")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Detections")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Numeric summary of owl_df")
    st.write(owl_df.describe())

    num_cols_all = owl_df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols_all) > 2:
        st.subheader("Correlation heatmap (numeric features)")
        corr = owl_df[num_cols_all].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(num_cols_all)))
        ax.set_xticklabels(num_cols_all, rotation=90)
        ax.set_yticks(range(len(num_cols_all)))
        ax.set_yticklabels(num_cols_all)
        fig.colorbar(im)
        st.pyplot(fig)

# =========================
# MODELING TAB
# =========================
with tabs[2]:
    st.subheader("Regression: Predict stay_duration_days")

    # Step 9 – Preparing data for regression
    y_reg = owl_df["stay_duration_days"]
    drop_cols = ["motusTagID", "stay_duration_days", "ResidencyType_true"]
    X = owl_df.drop(columns=[c for c in drop_cols if c in owl_df.columns], errors="ignore")

    # numeric only
    X = X.select_dtypes(include=np.number).copy()
    X = X.dropna(axis=1, how="all")

    # median impute
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = safe_split(X_imp, y_reg, task="reg")

    # Step 10 – Training multiple regression models
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
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rows.append({"Model": name, "R2": r2, "RMSE": rmse})
        pred_store[name] = pred

    reg_results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    st.dataframe(reg_results)

    best_reg_name = reg_results.iloc[0]["Model"]
    best_reg = reg_models[best_reg_name]
    st.write(f" Best Regression Model: **{best_reg_name}**")

    # Step 11 – Predicting stay duration for all owls with best model
    best_reg.fit(X_imp, y_reg)
    owl_df["predicted_stay_days"] = best_reg.predict(X_imp)

    st.subheader("Regression: True vs Predicted Stay Duration")
    fig, ax = plt.subplots(figsize=(7, 6))
    best_pred = pred_store[best_reg_name]
    ax.scatter(y_test, best_pred, alpha=0.7, edgecolors="black")
    ax.set_xlabel("True Stay Duration (days)")
    ax.set_ylabel("Predicted Stay Duration (days)")
    ax.set_title(f"Regression: True vs Predicted Stay Duration\nModel = {best_reg_name}")
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Classification: Predict ResidencyType_true")

    # Step 12 – Encoding residency labels
    X_cls = X_imp.copy()
    y_cls = owl_df["ResidencyType_true"].astype(str)

    le = LabelEncoder()
    y_cls_enc = le.fit_transform(y_cls)

    Xc_train, Xc_test, yc_train, yc_test = safe_split(X_cls, y_cls_enc, task="cls")

    # Step 13 – Hyperparameter tuning for classification
    candidates = {
        "LogReg": {
            "estimator": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
            ]),
            "params": {
                "clf__C": np.logspace(-2, 2, 12),
                "clf__solver": ["lbfgs", "liblinear"]
            }
        },
        "SVC": {
            "estimator": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", SVC(class_weight="balanced"))
            ]),
            "params": {
                "clf__C": np.logspace(-2, 2, 8),
                "clf__kernel": ["rbf", "poly"]
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample", random_state=42
            ),
            "params": {
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 6, 12, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [200, 400],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4]
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
    st.write(f"Best Classification Model: **{best_cls_name}**")

    # Step 14 – Classification report + confusion matrix
    yc_pred = best_cls.predict(Xc_test)

    all_labels = np.arange(len(le.classes_))  # [0,1,2] for Vagrant, Migrant, Resident

    st.text("Classification Report:")
    st.text(classification_report(
        yc_test,
        yc_pred,
        labels=all_labels,
        target_names=le.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(yc_test, yc_pred, labels=all_labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_)
    ax.set_yticks(range(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {best_cls_name}")
    fig.colorbar(im)
    st.pyplot(fig)

    # store for XAI tab
    st.session_state["best_reg"] = best_reg
    st.session_state["best_cls"] = best_cls
    st.session_state["X_imp"] = X_imp
    st.session_state["y_reg"] = y_reg
    st.session_state["Xc_test"] = Xc_test
    st.session_state["yc_test"] = yc_test

# =========================
# XAI TAB
# =========================
with tabs[3]:
    st.subheader("Explainability (XAI)")

    if "best_reg" not in st.session_state:
        st.info("Train models first in the Modeling tab.")
        st.stop()

    best_reg = st.session_state["best_reg"]
    best_cls = st.session_state["best_cls"]
    X_imp = st.session_state["X_imp"]
    y_reg = st.session_state["y_reg"]
    Xc_test = st.session_state["Xc_test"]
    yc_test = st.session_state["yc_test"]

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
