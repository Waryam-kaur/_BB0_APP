import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVR, SVC


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="BBO Owl Migration MVP", page_icon="🦉", layout="wide")
st.title("🦉 BBO Owl Migration MVP")
st.write("EDA + Feature Engineering + Modeling (Regression & Classification)")

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header(" Upload your files")

combined_file = st.sidebar.file_uploader(
    "Upload combined detections CSV",
    type=["csv"]
)

old_meta_file = st.sidebar.file_uploader(
    "Upload old metadata CSV (clean_df_selected.csv) – optional",
    type=["csv"]
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Full Pipeline ")


# =====================================================
# HELPERS (from your notebooks)
# =====================================================
def safe_mean(s): return pd.to_numeric(s, errors="coerce").mean()
def safe_std(s):  return pd.to_numeric(s, errors="coerce").std()

def cap_outliers_iqr(df, numeric_cols):
    df_capped = df.copy()
    for col in numeric_cols:
        if col in df_capped.columns:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - 1.5 * IQR
            upper_cap = Q3 + 1.5 * IQR
            df_capped[col] = df_capped[col].clip(lower_cap, upper_cap)
    return df_capped


@st.cache_data
def load_csv(file):
    return pd.read_csv(file, low_memory=False)


@st.cache_data
def prepare_pipeline(df_new, df_old=None):
    # ---------------------------
    # 1) Standardize motusTagID
    # ---------------------------
    possible_id_cols = [c for c in df_new.columns if "motus" in c.lower() and "tag" in c.lower()]
    possible_id_cols += [c for c in df_new.columns if c.lower() in ["tag_id","tagid","motustagid_sheet"]]

    if len(possible_id_cols) == 0:
        raise ValueError("No motusTagID column found in combined detections file.")

    id_col = possible_id_cols[0]
    df_new = df_new.rename(columns={id_col: "motusTagID"})
    df_new["motusTagID"] = pd.to_numeric(df_new["motusTagID"], errors="coerce")

    # ---------------------------
    # 2) Create DATETIME (same as modeling notebook)
    # ---------------------------
    df_new = df_new.copy()

    if "DATETIME" in df_new.columns:
        df_new["DATETIME"] = pd.to_datetime(df_new["DATETIME"], errors="coerce")
    else:
        if "DATE" in df_new.columns:
            df_new["DATE"] = pd.to_datetime(df_new["DATE"], errors="coerce")
        else:
            df_new["DATE"] = pd.NaT

        if "TIME" in df_new.columns:
            df_new["TIME_clean"] = df_new["TIME"].astype(str).str.extract(r"(\d{1,2}:\d{2}:\d{2})")[0]
            df_new["TIME_clean"] = pd.to_timedelta(df_new["TIME_clean"], errors="coerce")
        else:
            df_new["TIME_clean"] = pd.to_timedelta(np.nan)

        df_new["DATETIME"] = df_new["DATE"] + df_new["TIME_clean"]

    if "DATETIME" in df_new.columns:
        df_new["hour"] = df_new["DATETIME"].dt.hour

    # ---------------------------
    # 3) True stay duration per owl
    # ---------------------------
    first_det = df_new.groupby("motusTagID")["DATETIME"].min()
    last_det = df_new.groupby("motusTagID")["DATETIME"].max()

    stay_true = (last_det - first_det).dt.total_seconds() / (3600 * 24)
    stay_true = stay_true.clip(lower=0)

    stay_true = stay_true.reset_index()
    stay_true.columns = ["motusTagID", "stay_duration_days"]

    # ---------------------------
    # 4) Owl-level aggregated features
    # ---------------------------
    num_cols = ["snr","sigsd","freq","freqsd","slop","burstSlop","antBearing","port","nodeNum","runLen","hour"]
    num_cols = [c for c in num_cols if c in df_new.columns]

    agg_dict = {"detections_count": ("motusTagID", "size")}
    for c in num_cols:
        agg_dict[f"{c}_mean"] = (c, safe_mean)
        agg_dict[f"{c}_std"] = (c, safe_std)

    owl_features = df_new.groupby("motusTagID").agg(**agg_dict).reset_index()

    # ---------------------------
    # 5) Merge old metadata (optional)
    # ---------------------------
    meta_one = None
    if df_old is not None:
        df_old = df_old.copy()
        possible_old_ids = [c for c in df_old.columns if "motus" in c.lower() and "tag" in c.lower()]
        possible_old_ids += [c for c in df_old.columns if c.lower() in ["tag_id","tagid","motustagid_sheet"]]
        if len(possible_old_ids) > 0:
            old_id_col = possible_old_ids[0]
            df_old.rename(columns={old_id_col: "motusTagID"}, inplace=True)
            df_old["motusTagID"] = pd.to_numeric(df_old["motusTagID"], errors="coerce")
            meta_one = df_old.groupby("motusTagID").first().reset_index()

    owl_df = owl_features.merge(stay_true, on="motusTagID", how="left")
    if meta_one is not None:
        owl_df = owl_df.merge(meta_one, on="motusTagID", how="left", suffixes=("", "_meta"))

    # ---------------------------
    # 6) ResidencyType_true bins (same as your notebook)
    # ---------------------------
    bins = [0, 3, 7, np.inf]
    labels = ["Vagrant", "Migrant", "Resident"]

    owl_df["ResidencyType_true"] = pd.cut(
        owl_df["stay_duration_days"],
        bins=bins, labels=labels, include_lowest=True
    )

    return df_new, owl_df


# =====================================================
# RUN PIPELINE
# =====================================================
if combined_file is None:
    st.warning("Upload your combined detections CSV to start.")
    st.stop()

df_new = load_csv(combined_file)
df_old = load_csv(old_meta_file) if old_meta_file else None

if not run_btn:
    st.info("Upload files and click **Run Full Pipeline ** in sidebar.")
    st.stop()

try:
    df_new, owl_df = prepare_pipeline(df_new, df_old)
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "🛠️ Feature Engineering",
    "🤖 Modeling",
    "📌 Final Results"
])

# =====================================================
# TAB 1: EDA
# =====================================================
with tab1:
    st.header(" Exploratory Data Analysis")

    st.subheader("Combined detections preview")
    st.dataframe(df_new.head(30))
    st.write("Shape:", df_new.shape)

    st.markdown("---")

    st.subheader("Numeric Distributions")
    num_cols = ["snr", "sig", "noise", "freq"]
    for col in num_cols:
        if col in df_new.columns:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df_new[col], kde=True, bins=50, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

    if "hour" in df_new.columns:
        st.subheader("Hourly Detection Frequency")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df_new["hour"], bins=24, ax=ax)
        ax.set_title("Hourly detections")
        ax.set_xlabel("Hour of day")
        st.pyplot(fig)


# =====================================================
# TAB 2: FEATURE ENGINEERING
# =====================================================
with tab2:
    st.header(" Feature Engineering")

    st.subheader("Outlier capping (IQR)")
    numeric_cols = ["snr", "sig", "noise", "freq"]
    df_capped = cap_outliers_iqr(df_new, numeric_cols)

    for col in numeric_cols:
        if col in df_new.columns:
            fig, ax = plt.subplots(1,2, figsize=(10,3))
            sns.boxplot(x=df_new[col], ax=ax[0])
            ax[0].set_title(f"Before: {col}")
            sns.boxplot(x=df_capped[col], ax=ax[1])
            ax[1].set_title(f"After: {col}")
            st.pyplot(fig)

    st.markdown("---")

    st.subheader("Owl-level engineered dataset")
    st.dataframe(owl_df.head(25))
    st.write("Shape:", owl_df.shape)

    st.write("ResidencyType counts:")
    st.write(owl_df["ResidencyType_true"].value_counts(dropna=False))


# =====================================================
# TAB 3: MODELING
# =====================================================
with tab3:
    st.header(" Modeling")

    # --------------------------
    # A) REGRESSION
    # --------------------------
    st.subheader("A) Regression — predict stay_duration_days")

    y_reg = owl_df["stay_duration_days"]

    X = owl_df.drop(columns=["stay_duration_days", "ResidencyType_true"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.dropna(axis=1, how="all")

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y_reg, test_size=0.2, random_state=42
    )

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
    best_pred = pred_store[best_reg_name]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y_test, best_pred, alpha=0.7)
    ax.set_xlabel("True stay duration (days)")
    ax.set_ylabel("Predicted stay duration (days)")
    ax.set_title(f"Best Regression Model: {best_reg_name}")
    st.pyplot(fig)

    # --------------------------
    # B) CLASSIFICATION
    # --------------------------
    st.markdown("---")
    st.subheader("B) Classification — predict ResidencyType_true")

    y_cls = owl_df["ResidencyType_true"].astype(str)
    le = LabelEncoder()
    y_cls_enc = le.fit_transform(y_cls)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_imp, y_cls_enc, test_size=0.2, random_state=42, stratify=y_cls_enc
    )

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
                "clf__C": np.logspace(-2,2,12),
                "clf__gamma": ["scale","auto"]
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
            "params": {
                "n_estimators": [200,400,800],
                "max_depth": [None,6,12,20],
                "min_samples_split": [2,5,10]
            }
        }
    }

    search_results = []
    best_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, cfg in candidates.items():
        search = RandomizedSearchCV(
            cfg["estimator"], cfg["params"],
            n_iter=12, scoring="f1_macro",
            cv=cv, random_state=42, n_jobs=-1
        )
        search.fit(Xc_train, yc_train)

        best_models[name] = search.best_estimator_
        pred = best_models[name].predict(Xc_test)

        search_results.append({
            "Model": name,
            "BestCV_F1": search.best_score_,
            "Test_F1": f1_score(yc_test, pred, average="macro"),
            "Test_Acc": accuracy_score(yc_test, pred),
            "BestParams": search.best_params_
        })

    cls_results = pd.DataFrame(search_results).sort_values("Test_F1", ascending=False)
    st.dataframe(cls_results)

    best_cls_name = cls_results.iloc[0]["Model"]
    best_cls = best_models[best_cls_name]
    yc_pred = best_cls.predict(Xc_test)

    st.text("Classification Report:")
    st.text(classification_report(yc_test, yc_pred, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(yc_test, yc_pred, labels=np.unique(y_cls_enc))

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {best_cls_name}")
    st.pyplot(fig)


# =====================================================
# TAB 4: FINAL RESULTS
# =====================================================
with tab4:
    st.header(" Final Results (Simple View)")

    st.write(f" Total owls analyzed: **{owl_df['motusTagID'].nunique()}**")
    st.write(f" Best regression model: **{best_reg_name}**")
    st.write(f" Best classification model: **{best_cls_name}**")

    st.markdown("---")
    st.subheader("Residency Type Summary")
    st.bar_chart(owl_df["ResidencyType_true"].value_counts())

    st.markdown("---")
    st.subheader("Download owl-level dataset")
    st.download_button(
        "Download owl_df as CSV",
        owl_df.to_csv(index=False).encode("utf-8"),
        file_name="owl_level_dataset.csv",
        mime="text/csv"
    )
