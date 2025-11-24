import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC


# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="BBO Owl Migration MVP", page_icon="🦉", layout="wide")

st.title("🦉 BBO Owl Migration MVP")
st.caption("EDA + Feature Engineering + Modeling (Regression & Classification)")

# --------------------------
# HELPERS
# --------------------------
@st.cache_data(show_spinner=False)
def load_detections_excel(uploaded_xlsx: BytesIO):
    """Reads all sheets from Excel and stacks them into one detections df."""
    xls = pd.ExcelFile(uploaded_xlsx)
    sheet_names = xls.sheet_names

    all_dfs = []
    for sheet in sheet_names:
        try:
            temp = pd.read_excel(xls, sheet_name=sheet)
            if temp.shape[0] == 0:
                continue
            temp["motusTagID_sheet"] = sheet  # keep sheet id
            all_dfs.append(temp)
        except Exception:
            continue

    df = pd.concat(all_dfs, ignore_index=True)
    return df


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback by partial matches
    low_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low_cols:
            return low_cols[c.lower()]
    return None


def make_datetime(df):
    """Create a unified datetime column from DATE/TIME or tsCorrected or ts."""
    date_col = find_col(df, ["DATE", "date"])
    time_col = find_col(df, ["TIME", "time"])
    ts_col = find_col(df, ["tsCorrected", "ts_corrected", "ts"])

    if date_col and time_col:
        dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    elif ts_col:
        # tsCorrected looks like unix seconds in your screenshot
        dt = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    else:
        dt = pd.to_datetime(df.index, errors="coerce")

    df["datetime"] = dt
    return df


@st.cache_data(show_spinner=False)
def build_owl_features(df_det):
    """Aggregate detections to owl-level features."""
    df_det = make_datetime(df_det)

    id_col = find_col(df_det, ["motusTagID", "motusTagID_sheet", "tag_id", "tagid"])
    if id_col is None:
        # use sheet name if no id inside
        id_col = "motusTagID_sheet"

    df_det[id_col] = pd.to_numeric(df_det[id_col], errors="coerce")

    # stay duration true
    stay = df_det.groupby(id_col)["datetime"].agg(["min", "max"]).reset_index()
    stay["stay_duration_days"] = (stay["max"] - stay["min"]).dt.total_seconds() / (24 * 3600)

    # numeric columns to aggregate
    numeric_cols = df_det.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col]

    agg_dict = {}
    for c in numeric_cols:
        agg_dict[c+"_mean"] = (c, "mean")
        agg_dict[c+"_std"] = (c, "std")
        agg_dict[c+"_min"] = (c, "min")
        agg_dict[c+"_max"] = (c, "max")

    agg_dict["n_detections"] = (numeric_cols[0], "count") if numeric_cols else ("datetime", "count")

    owl_features = df_det.groupby(id_col).agg(**agg_dict).reset_index()
    owl_features = owl_features.merge(stay[[id_col, "stay_duration_days"]], on=id_col, how="left")
    owl_features.rename(columns={id_col: "motusTagID"}, inplace=True)

    return owl_features


@st.cache_data(show_spinner=False)
def load_old_metadata(uploaded_csv):
    df_old = pd.read_csv(uploaded_csv, low_memory=False)
    id_old = find_col(df_old, ["motusTagID", "motusTagID_sheet", "tag_id", "tagid"])
    if id_old:
        df_old.rename(columns={id_old: "motusTagID"}, inplace=True)
    df_old["motusTagID"] = pd.to_numeric(df_old["motusTagID"], errors="coerce")
    return df_old


def classify_stay(stay_days, short_thr, long_thr):
    if stay_days <= short_thr:
        return "vagrant"
    elif stay_days <= long_thr:
        return "migrant"
    else:
        return "resident"


def preprocess_for_ml(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder",  LabelEncoderWrapper())  # custom wrapper below
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return X, y, pre


class LabelEncoderWrapper:
    """ColumnTransformer-friendly label encoding for multiple categorical columns."""
    def fit(self, X, y=None):
        self.encoders_ = []
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            le = LabelEncoder()
            le.fit(X_df[col].astype(str))
            self.encoders_.append(le)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        out = []
        for i, col in enumerate(X_df.columns):
            le = self.encoders_[i]
            out.append(le.transform(X_df[col].astype(str)))
        return np.vstack(out).T


# --------------------------
# SIDEBAR: MODE & INPUTS
# --------------------------
st.sidebar.header("Run Mode")

demo_mode = st.sidebar.checkbox("Use DEMO data (no restricted upload)", value=True)

with st.sidebar.expander("Private Data Upload (restricted)", expanded=not demo_mode):
    xlsx_up = st.file_uploader("Upload SawWhets detections Excel (.xlsx)", type=["xlsx"])
    csv_up  = st.file_uploader("Upload old metadata clean_df_selected.csv", type=["csv"])

short_thr = st.sidebar.slider("Short stay threshold (days)", 0.5, 5.0, 2.0, 0.5)
long_thr  = st.sidebar.slider("Long stay threshold (days)", 3.0, 20.0, 7.0, 1.0)

run_btn = st.sidebar.button("Run Full Pipeline")


# --------------------------
# DATA SOURCE
# --------------------------
def make_demo_data():
    # tiny demo dataset so app opens with results
    np.random.seed(0)
    demo_det = pd.DataFrame({
        "motusTagID": np.repeat([80830,80831,80832], [40,35,30]),
        "tsCorrected": np.concatenate([
            np.linspace(1_697_000_000, 1_697_200_000, 40),
            np.linspace(1_697_100_000, 1_697_160_000, 35),
            np.linspace(1_697_300_000, 1_697_320_000, 30),
        ]),
        "snr": np.random.normal(0.35, 0.05, 105),
        "sig": np.random.normal(-50, 3, 105),
        "noise": np.random.normal(-80, 2, 105),
        "freq": np.random.normal(3.8, 0.1, 105)
    })
    demo_old = pd.DataFrame({
        "motusTagID": [80830,80831,80832],
        "age": ["HY","AHY","HY"],
        "sex": ["F","M","U"],
        "weight": [83, 88, 79],
        "wing": [140, 142, 138]
    })
    return demo_det, demo_old


if demo_mode:
    df_det, df_old = make_demo_data()
else:
    if xlsx_up and csv_up:
        df_det = load_detections_excel(xlsx_up)
        df_old = load_old_metadata(csv_up)
    else:
        st.info("Upload both restricted files OR turn on DEMO mode.")
        st.stop()


# --------------------------
# RUN PIPELINE
# --------------------------
if run_btn or demo_mode:
    with st.spinner("Building owl-level features and merging datasets..."):
        owl_features = build_owl_features(df_det)
        combined = owl_features.merge(df_old, on="motusTagID", how="left")

        combined["stay_class"] = combined["stay_duration_days"].apply(
            lambda x: classify_stay(x, short_thr, long_thr)
        )

    # --------------------------
    # TABS UI
    # --------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Home", "EDA", "Feature Engineering", "Modeling"])

    with tab1:
        st.subheader("Dataset Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Detections rows", df_det.shape[0])
        c2.metric("Unique owls", owl_features.shape[0])
        c3.metric("Final features rows", combined.shape[0])
        st.write("Final merged dataset preview:")
        st.dataframe(combined.head(20))

    with tab2:
        st.subheader("EDA")
        st.write("Stay duration distribution (days)")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(combined["stay_duration_days"], bins=15, kde=True, ax=ax)
        ax.set_xlabel("Stay Duration (days)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.write("Top numeric correlations")
        num_df = combined.select_dtypes(include=[np.number])
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig2, ax2 = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        else:
            st.info("Not enough numeric columns to plot correlation heatmap.")

    with tab3:
        st.subheader("Feature Engineering Output")
        st.write("Owl-level features created from detections:")
        st.dataframe(owl_features.head(20))

        st.write("Stay class counts:")
        st.bar_chart(combined["stay_class"].value_counts())

    with tab4:
        st.subheader("Regression (predict stay duration)")
        reg_df = combined.dropna(subset=["stay_duration_days"]).copy()

        Xr, yr, pre_r = preprocess_for_ml(reg_df, "stay_duration_days")
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            Xr, yr, test_size=0.25, random_state=42
        )

        regressors = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "SVR": SVR()
        }

        reg_results = []
        best_name, best_model, best_rmse = None, None, np.inf

        for name, model in regressors.items():
            pipe = Pipeline([("preprocess", pre_r), ("model", model)])
            pipe.fit(Xr_train, yr_train)
            pred = pipe.predict(Xr_test)
            rmse = np.sqrt(mean_squared_error(yr_test, pred))
            r2 = r2_score(yr_test, pred)
            reg_results.append((name, rmse, r2))
            if rmse < best_rmse:
                best_rmse, best_name, best_model = rmse, name, pipe

        reg_table = pd.DataFrame(reg_results, columns=["Model", "RMSE", "R2"]).sort_values("RMSE")
        st.dataframe(reg_table, use_container_width=True)
        st.success(f"Best regression model: **{best_name}** (RMSE={best_rmse:.2f})")

        st.write("True vs Predicted (best model)")
        best_pred = best_model.predict(Xr_test)
        fig3, ax3 = plt.subplots(figsize=(6,5))
        ax3.scatter(yr_test, best_pred)
        ax3.set_xlabel("True stay days")
        ax3.set_ylabel("Predicted stay days")
        ax3.grid(True)
        st.pyplot(fig3)

        st.divider()
        st.subheader("Classification (resident / migrant / vagrant)")
        cls_df = combined.dropna(subset=["stay_class"]).copy()
        Xc, yc, pre_c = preprocess_for_ml(cls_df, "stay_class")

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.25, random_state=42, stratify=yc
        )

        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }

        cls_results = []
        bestc_name, bestc_model, best_f1 = None, None, -1

        for name, model in classifiers.items():
            pipe = Pipeline([("preprocess", pre_c), ("model", model)])
            pipe.fit(Xc_train, yc_train)
            pred = pipe.predict(Xc_test)
            acc = accuracy_score(yc_test, pred)
            f1 = f1_score(yc_test, pred, average="weighted")
            cls_results.append((name, acc, f1))
            if f1 > best_f1:
                best_f1, bestc_name, bestc_model = f1, name, pipe

        cls_table = pd.DataFrame(cls_results, columns=["Model", "Accuracy", "F1"]).sort_values("F1", ascending=False)
        st.dataframe(cls_table, use_container_width=True)
        st.success(f"Best classification model: **{bestc_name}** (F1={best_f1:.2f})")

        bestc_pred = bestc_model.predict(Xc_test)
        st.text("Classification report:")
        st.code(classification_report(yc_test, bestc_pred))

        cm = confusion_matrix(yc_test, bestc_pred, labels=np.unique(yc))
        fig4, ax4 = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(yc), yticklabels=np.unique(yc), ax=ax4)
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("True")
        st.pyplot(fig4)

    # download final dataset (without storing restricted files)
    st.sidebar.download_button(
        "Download final merged dataset (CSV)",
        data=combined.to_csv(index=False).encode("utf-8"),
        file_name="FULL_DATASET_from_app.csv",
        mime="text/csv"
    )
