import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="BBO Owl Migration MVP", page_icon="🦉", layout="wide")
st.title("🦉 BBO Owl Migration MVP")
st.caption("EDA + Feature Engineering + Modeling (Regression & Classification)")


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
@st.cache_data(show_spinner=False)
def load_detections_excel(excel_path):
    """Read all sheets, clean, and combine into one detections df."""
    xls = pd.ExcelFile(excel_path)
    all_sheets = []
    for sh in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sh)

        # drop unnamed junk cols
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # standardize column names
        df.columns = df.columns.astype(str)

        all_sheets.append(df)

    combined = pd.concat(all_sheets, ignore_index=True)

    # Ensure datetime
    if "DATE" in combined.columns:
        combined["DATE"] = pd.to_datetime(combined["DATE"], errors="coerce")

    # motusTagID as string
    if "motusTagID" in combined.columns:
        combined["motusTagID"] = combined["motusTagID"].astype(str)

    return combined


@st.cache_data(show_spinner=False)
def load_demo_data():
    """
    Load NON-restricted demo data from repo.
    You must add:
      demo_detections.csv
      demo_metadata.csv
    """
    det_path = "demo_detections.csv"
    meta_path = "demo_metadata.csv"

    if not (os.path.exists(det_path) and os.path.exists(meta_path)):
        st.error(
            "Demo files not found in repo.\n\n"
            "Add these NON-restricted files to GitHub:\n"
            "- demo_detections.csv\n"
            "- demo_metadata.csv"
        )
        return None, None

    det = pd.read_csv(det_path)
    meta = pd.read_csv(meta_path)

    if "DATE" in det.columns:
        det["DATE"] = pd.to_datetime(det["DATE"], errors="coerce")
    det["motusTagID"] = det["motusTagID"].astype(str)

    if "tag_id" in meta.columns:
        meta["tag_id"] = meta["tag_id"].astype(str)

    return det, meta


def compute_stay_duration(detections_df):
    """Aggregate per owl and compute stay duration + basic detection stats."""
    df = detections_df.copy()

    if "motusTagID" not in df.columns:
        st.error("motusTagID column missing in detections file.")
        return None

    # compute stay duration by DATE min/max
    agg_dict = {
        "DATE": ["min", "max"],
        "motusTagID": "count"
    }

    # add numeric means if present
    numeric_candidates = [
        "snr", "sig", "sigsd", "noise", "freq", "freqsd",
        "slop", "burstSlop", "runLen", "nodeNum", "antBearing"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            agg_dict[c] = "mean"

    owl = df.groupby("motusTagID").agg(agg_dict)
    owl.columns = ["_".join(col).strip("_") for col in owl.columns]

    owl = owl.rename(columns={"motusTagID_count": "num_detections"})
    owl["stay_duration_days"] = (
        owl["DATE_max"] - owl["DATE_min"]
    ).dt.total_seconds() / (24 * 3600)

    owl = owl.reset_index()
    return owl


def make_stay_category(days, short_th=2, long_th=7):
    if pd.isna(days):
        return np.nan
    if days < short_th:
        return "short"
    elif days < long_th:
        return "medium"
    else:
        return "long"


def build_pipelines(num_cols, cat_cols):
    # preprocessing
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    reg_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    cls_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    reg_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", reg_model)
    ])

    cls_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", cls_model)
    ])

    return reg_pipe, cls_pipe


# -----------------------------
# SIDEBAR: MODE + THRESHOLDS
# -----------------------------
st.sidebar.header("Run Mode")

use_demo = st.sidebar.checkbox("Use DEMO data (no restricted upload)", value=True)

short_th = st.sidebar.slider("Short stay threshold (days)", 0.5, 5.0, 2.0, 0.5)
long_th  = st.sidebar.slider("Long stay threshold (days)", 3.0, 15.0, 7.0, 1.0)


# -----------------------------
# LOAD DATA (AUTO)
# -----------------------------
detections_df = None
meta_df = None

if use_demo:
    detections_df, meta_df = load_demo_data()
else:
    # local-only auto load (no upload)
    excel_local = "SawWhets_June3_2024-2.xlsx"
    meta_local = "clean_df_selected.csv"

    if os.path.exists(excel_local) and os.path.exists(meta_local):
        detections_df = load_detections_excel(excel_local)
        meta_df = pd.read_csv(meta_local)
        meta_df["tag_id"] = meta_df["tag_id"].astype(str)
    else:
        st.warning(
            "Local files not found.\n\n"
            "Put these in the same folder as streamlit_app.py:\n"
            "- SawWhets_June3_2024-2.xlsx\n"
            "- clean_df_selected.csv\n\n"
            "Or switch back to DEMO mode."
        )


if detections_df is None or meta_df is None:
    st.stop()


# -----------------------------
# PIPELINE
# -----------------------------
st.subheader("1) Combine + Compute Stay Duration")

owl_df = compute_stay_duration(detections_df)
st.write("Per-owl aggregated detections:")
st.dataframe(owl_df.head(10), use_container_width=True)

st.subheader("2) Merge with Old Metadata")

# merge
if "tag_id" not in meta_df.columns:
    st.error("clean_df_selected.csv must contain tag_id column.")
    st.stop()

final_df = owl_df.merge(
    meta_df,
    left_on="motusTagID",
    right_on="tag_id",
    how="left"
)

# stay category
final_df["stay_category"] = final_df["stay_duration_days"].apply(
    lambda x: make_stay_category(x, short_th, long_th)
)

st.write("Final merged dataset preview:")
st.dataframe(final_df.head(10), use_container_width=True)


# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["EDA", "Feature Engineering", "Modeling"])


with tab1:
    st.header("EDA")

    colA, colB = st.columns(2)

    with colA:
        st.write("Stay duration distribution")
        fig = plt.figure()
        sns.histplot(final_df["stay_duration_days"].dropna(), bins=30, kde=True)
        plt.xlabel("Stay Duration (days)")
        st.pyplot(fig)

    with colB:
        st.write("Stay category counts")
        fig = plt.figure()
        sns.countplot(data=final_df, x="stay_category",
                      order=["short", "medium", "long"])
        st.pyplot(fig)

    st.write("Top 10 owls by detections")
    st.dataframe(
        final_df.sort_values("num_detections", ascending=False)
                [["motusTagID","num_detections","stay_duration_days","stay_category"]]
                .head(10),
        use_container_width=True
    )


with tab2:
    st.header("Feature Engineering")

    # choose usable cols
    target_reg = "stay_duration_days"
    target_cls = "stay_category"

    drop_cols = [
        "DATE_min", "DATE_max", "tag_id"
    ]
    features_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

    # separate X/y
    X = features_df.drop(columns=[target_reg, target_cls])
    y_reg = features_df[target_reg]
    y_cls = features_df[target_cls]

    st.write("Features used for modeling:")
    st.write(list(X.columns))

    st.write("Missing values summary:")
    st.dataframe(X.isna().sum().sort_values(ascending=False).head(20))


with tab3:
    st.header("Modeling (Regression + Classification)")

    target_reg = "stay_duration_days"
    target_cls = "stay_category"

    drop_cols = ["DATE_min", "DATE_max", "tag_id"]
    features_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

    X = features_df.drop(columns=[target_reg, target_cls])
    y_reg = features_df[target_reg]
    y_cls = features_df[target_cls]

    # detect column types
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    st.write("Numeric cols:", num_cols)
    st.write("Categorical cols:", cat_cols)

    # splits
    X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )

    reg_pipe, cls_pipe = build_pipelines(num_cols, cat_cols)

    # fit regression
    st.subheader("Regression: Predict Stay Duration (days)")
    reg_pipe.fit(X_train, yreg_train)
    pred_reg = reg_pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(yreg_test, pred_reg))
    r2 = r2_score(yreg_test, pred_reg)

    st.write(f"RMSE: **{rmse:.3f}**")
    st.write(f"R² Score: **{r2:.3f}**")

    fig = plt.figure()
    plt.scatter(yreg_test, pred_reg, alpha=0.6)
    plt.xlabel("Actual stay duration")
    plt.ylabel("Predicted stay duration")
    plt.title("Actual vs Predicted (Regression)")
    st.pyplot(fig)

    # fit classification
    st.subheader("Classification: Predict Stay Category")
    cls_pipe.fit(X_train, ycls_train)
    pred_cls = cls_pipe.predict(X_test)

    acc = accuracy_score(ycls_test, pred_cls)
    st.write(f"Accuracy: **{acc:.3f}**")

    st.text("Classification Report:")
    st.text(classification_report(ycls_test, pred_cls))

    cm = confusion_matrix(ycls_test, pred_cls, labels=["short","medium","long"])
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["short","medium","long"],
                yticklabels=["short","medium","long"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.success("Pipeline complete ")
