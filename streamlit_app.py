import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    r2_score, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

st.set_page_config(
    page_title="BBO Owl Migration MVP",
    page_icon="🦉",
    layout="wide"
)

# =========================================================
# Helper functions (same logic as notebook)
# =========================================================

def standardize_tag_id(df, col):
    """Convert tag column to numeric string like notebook."""
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).copy()
    df[col] = df[col].astype(int).astype(str)
    return df

def load_detections_excel(xlsx_file):
    """
    Loads SawWhets excel detections file.
    Notebook logic:
    - parse DATE + TIME
    - create DATETIME
    """
    xls = pd.ExcelFile(xlsx_file)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xlsx_file, sheet_name=sheet)

    # expected columns from MOTUS detections
    # DATE / TIME should exist in your file
    if "DATE" in df.columns and "TIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            errors="coerce"
        )
    elif "tsCorrected" in df.columns:
        # fallback if DATETIME not there
        df["DATETIME"] = pd.to_datetime(df["tsCorrected"], unit="s", errors="coerce")
    else:
        st.warning("No DATE+TIME or tsCorrected found. DATETIME may be missing.")

    # standardize motusTagID
    tag_col = "motusTagID" if "motusTagID" in df.columns else "tag_id"
    df = standardize_tag_id(df, tag_col)
    df.rename(columns={tag_col: "motusTagID"}, inplace=True)

    df = df.dropna(subset=["DATETIME"])
    return df

def load_metadata_csv(csv_file):
    """
    Loads clean_df_selected.csv (old metadata).
    Notebook logic:
    - standardize tag id
    """
    meta = pd.read_csv(csv_file)

    # find tag col name
    possible = [c for c in meta.columns if c.lower() in ["motustagid", "tag_id", "tagid", "nanotagid"]]
    if possible:
        tag_col = possible[0]
    else:
        tag_col = meta.columns[0]  # last fallback

    meta = standardize_tag_id(meta, tag_col)
    meta.rename(columns={tag_col: "motusTagID"}, inplace=True)
    return meta

def compute_stay_duration(detections_df):
    """
    Stay duration per owl = max(DATETIME) - min(DATETIME)
    same as notebook.
    """
    g = detections_df.groupby("motusTagID")["DATETIME"]
    stay = pd.DataFrame({
        "motusTagID": g.min().index,
        "first_seen": g.min().values,
        "last_seen": g.max().values
    })
    stay["stay_duration_days"] = (stay["last_seen"] - stay["first_seen"]).dt.total_seconds() / (60*60*24)
    return stay

def classify_stay(dur, short_thr=2, long_thr=7):
    """
    Proxy rule:
    short stay -> vagrant
    medium stay -> migrant
    long stay -> resident
    """
    if dur <= short_thr:
        return "short"
    elif dur <= long_thr:
        return "medium"
    else:
        return "long"

# =========================================================
# Sidebar: Data source selection
# =========================================================

st.sidebar.title("Run Mode")

mode = st.sidebar.radio(
    "Choose data source",
    ["Use DEMO data (no restricted upload)", "Google Drive / Local Path", "Private Manual Upload (restricted OK)"],
    index=2
)

short_thr = st.sidebar.slider("Short stay threshold (days)", 0.5, 5.0, 2.0, 0.5)
long_thr = st.sidebar.slider("Long stay threshold (days)", 5.0, 20.0, 7.0, 1.0)

detections_file = None
metadata_file = None

if mode == "Use DEMO data (no restricted upload)":
    st.sidebar.info("DEMO mode loads local non-restricted demo files if you add them later.")
    detections_path = "demo_detections.csv"
    metadata_path = "demo_metadata.csv"
    try:
        detections_file = detections_path
        metadata_file = metadata_path
    except:
        detections_file = None
        metadata_file = None

elif mode == "Google Drive / Local Path":
    st.sidebar.info("This mode is for running locally on your machine.")
    detections_path = st.sidebar.text_input(
        "Path to SawWhets_June3_2024-2.xlsx",
        value="SawWhets_June3_2024-2.xlsx"
    )
    metadata_path = st.sidebar.text_input(
        "Path to clean_df_selected.csv",
        value="clean_df_selected.csv"
    )
    detections_file = detections_path
    metadata_file = metadata_path

else:
    st.sidebar.warning(
        "Private upload: your data is NOT saved anywhere.\n"
        "Only you see it during this run."
    )
    detections_file = st.sidebar.file_uploader(
        "Upload SawWhets detections Excel (.xlsx)",
        type=["xlsx"]
    )
    metadata_file = st.sidebar.file_uploader(
        "Upload clean_df_selected.csv (old metadata)",
        type=["csv"]
    )

# =========================================================
# Main: Load + Process
# =========================================================

st.title("🦉 BBO Owl Migration MVP")
st.caption("EDA + Feature Engineering + Modeling + XAI (Regression & Classification)")

@st.cache_data(show_spinner=False)
def build_final_df(detections_file, metadata_file, short_thr, long_thr):
    det = load_detections_excel(detections_file)
    meta = load_metadata_csv(metadata_file)

    stay = compute_stay_duration(det)
    final_df = stay.merge(meta, on="motusTagID", how="left")

    final_df["stay_category"] = final_df["stay_duration_days"].apply(
        lambda x: classify_stay(x, short_thr, long_thr)
    )
    final_df["migration_proxy"] = final_df["stay_category"].map({
        "short":"vagrant",
        "medium":"migrant",
        "long":"resident"
    })
    return det, meta, final_df

if detections_file is None or metadata_file is None:
    st.info("Upload / provide both files to run the pipeline.")
    st.stop()

try:
    detections_df, metadata_df, df = build_final_df(detections_file, metadata_file, short_thr, long_thr)
except Exception as e:
    st.error("Could not load/process your files.")
    st.exception(e)
    st.stop()

st.success("✅ Data loaded and pipeline finished automatically!")

# =========================================================
# Tabs
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "📊 EDA", "🤖 Modeling", "🧠 XAI"])

# ------------------------- Overview -------------------------
with tab1:
    st.subheader("Final Dataset (after combining old + new)")
    st.write(df.head(20))
    st.write("Shape:", df.shape)

    st.markdown("""
    **What this app does (same as notebook):**
    1. Loads MOTUS detections (new data) + clean_df_selected (old metadata)
    2. Standardizes motusTagID
    3. Creates DATETIME
    4. Computes stay duration per owl
    5. Merges duration with metadata
    6. Creates stay category + migration proxy (resident/migrant/vagrant)
    """)

# ------------------------- EDA -------------------------
with tab2:
    st.subheader("Stay Duration Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df["stay_duration_days"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Stay duration (days)")
    ax.set_ylabel("Owls count")
    st.pyplot(fig)

    st.subheader("Stay Category Counts")
    st.bar_chart(df["stay_category"].value_counts())

    st.subheader("Migration Proxy Counts")
    st.bar_chart(df["migration_proxy"].value_counts())

    st.subheader("Daily Owl Activity (detections per day)")
    detections_df["date_only"] = detections_df["DATETIME"].dt.date
    daily = detections_df.groupby("date_only")["motusTagID"].count().reset_index()

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(daily["date_only"], daily["motusTagID"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Detections")
    ax.set_title("Daily Detections at Station")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ------------------------- Modeling -------------------------
with tab3:
    st.subheader("Prepare features")

    # choose target
    target_task = st.radio(
        "Select task",
        ["Regression: Predict stay_duration_days", "Classification: Predict stay_category"],
        horizontal=True
    )

    # columns to ignore
    drop_cols = [c for c in ["first_seen","last_seen","DATETIME"] if c in df.columns]
    drop_cols += ["stay_category","migration_proxy","stay_duration_days"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if target_task.startswith("Regression"):
        y = df["stay_duration_days"]
    else:
        y = df["stay_category"]

    # split categorical / numeric
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object","category"]).columns.tolist()

    # preprocessing pipeline (FIXES your crash)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if target_task.startswith("Classification") else None
    )

    if target_task.startswith("Regression"):
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=300, random_state=42
            )
        }
        rows=[]
        preds={}
        for name, model in models.items():
            pipe = Pipeline([("prep", preprocess), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            r2 = r2_score(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            rows.append({"Model":name, "R2":r2, "RMSE":rmse})
            preds[name]=pred

        results = pd.DataFrame(rows).sort_values("R2", ascending=False)
        st.dataframe(results)

    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=300, random_state=42
            )
        }
        rows=[]
        best_pipe=None
        best_f1=-1

        for name, model in models.items():
            pipe = Pipeline([("prep", preprocess), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            rows.append({"Model":name, "Accuracy":acc, "F1":f1})

            if f1 > best_f1:
                best_f1=f1
                best_pipe=pipe

        results = pd.DataFrame(rows).sort_values("F1", ascending=False)
        st.dataframe(results)

        st.subheader("Classification Report (Best Model)")
        st.text(classification_report(y_test, best_pipe.predict(X_test)))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, best_pipe.predict(X_test), labels=["short","medium","long"])
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["short","medium","long"],
                    yticklabels=["short","medium","long"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ------------------------- XAI -------------------------
with tab4:
    st.subheader("Explainability (XAI)")

    task = st.selectbox(
        "Pick model to explain",
        ["Random Forest Regressor (stay duration)",
         "Random Forest Classifier (stay category)"]
    )

    if task.startswith("Random Forest Regressor"):
        y_reg = df["stay_duration_days"]
        X_reg = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        pipe = Pipeline([("prep", preprocess), ("model", rf)])
        pipe.fit(X_train, y_train)

        st.write("Feature importance (RF)")
        importances = pipe.named_steps["model"].feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]

        # get feature names after onehot
        ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols) if len(cat_cols)>0 else []
        feat_names = np.array(num_cols + list(cat_names))

        imp_df = pd.DataFrame({
            "feature": feat_names[top_idx],
            "importance": importances[top_idx]
        })
        st.dataframe(imp_df)

        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
        st.pyplot(fig)

        st.write("Permutation importance (test set)")
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42)
        perm_idx = np.argsort(perm.importances_mean)[::-1][:10]
        perm_df = pd.DataFrame({
            "feature": feat_names[perm_idx],
            "perm_importance": perm.importances_mean[perm_idx]
        })
        st.dataframe(perm_df)

    else:
        y_cls = df["stay_category"]
        X_cls = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )

        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        pipe = Pipeline([("prep", preprocess), ("model", rf)])
        pipe.fit(X_train, y_train)

        st.write("Feature importance (RF)")
        importances = pipe.named_steps["model"].feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]

        ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_cols) if len(cat_cols)>0 else []
        feat_names = np.array(num_cols + list(cat_names))

        imp_df = pd.DataFrame({
            "feature": feat_names[top_idx],
            "importance": importances[top_idx]
        })
        st.dataframe(imp_df)

        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
        st.pyplot(fig)

        st.write("Permutation importance (test set)")
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42)
        perm_idx = np.argsort(perm.importances_mean)[::-1][:10]
        perm_df = pd.DataFrame({
            "feature": feat_names[perm_idx],
            "perm_importance": perm.importances_mean[perm_idx]
        })
        st.dataframe(perm_df)

st.caption("✅ App finished. Restricted data is never stored; upload is private each run.")
