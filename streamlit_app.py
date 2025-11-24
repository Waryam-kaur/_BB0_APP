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
    RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
)
from sklearn.svm import SVR, SVC


# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(page_title="BBO Owl Migration MVP", page_icon="🦉", layout="wide")
st.title("🦉 BBO Owl Migration MVP")
st.caption("Restricted MOTUS data is NOT stored anywhere. You upload privately each run.")


# =====================================================
# SIDEBAR UPLOAD (ONLY)
# =====================================================
st.sidebar.header("🔒 Private Data Upload (Restricted)")

xlsx_up = st.sidebar.file_uploader(
    "Upload SawWhets detections Excel (.xlsx)",
    type=["xlsx"]
)

meta_up = st.sidebar.file_uploader(
    "Upload clean_df_selected.csv (old metadata)",
    type=["csv"]
)

if xlsx_up is None:
    st.info("Upload the detections Excel file in the sidebar to start.")
    st.stop()


# =====================================================
# HELPERS (same as notebooks)
# =====================================================
def safe_mean(s): return pd.to_numeric(s, errors="coerce").mean()
def safe_std(s):  return pd.to_numeric(s, errors="coerce").std()

def cap_outliers_iqr(df, cols):
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_out[col] = df_out[col].clip(lower, upper)
    return df_out

def build_datetime(df):
    df = df.copy()

    if "DATETIME" in df.columns:
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    else:
        # DATE
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        else:
            df["DATE"] = pd.NaT

        # TIME -> extract HH:MM:SS
        if "TIME" in df.columns:
            time_str = df["TIME"].astype(str).str.extract(r"(\d{1,2}:\d{2}:\d{2})")[0]
            df["TIME_clean"] = pd.to_timedelta(time_str, errors="coerce")
        else:
            df["TIME_clean"] = pd.to_timedelta(np.nan)

        df["DATETIME"] = df["DATE"] + df["TIME_clean"]

    # hour + TimeOfDay (simple, matches “Day vs Night” idea)
    df["hour"] = df["DATETIME"].dt.hour
    df["TimeOfDay"] = np.where((df["hour"] >= 18) | (df["hour"] < 6), "Night", "Day")
    return df


@st.cache_data
def load_detections_excel(uploaded_xlsx):
    """
    Exactly like your EDA notebook:
    - read all sheets
    - split sheet 80830 into 80830 + 80831
    - concatenate
    """
    xls = pd.ExcelFile(uploaded_xlsx)
    sheet_names = xls.sheet_names

    # mixed sheet
    if "80830" in sheet_names:
        df_80830 = pd.read_excel(xls, sheet_name="80830")
        df_owl_80830 = df_80830[df_80830["motusTagID"] == 80830].copy()
        df_owl_80831 = df_80830[df_80830["motusTagID"] == 80831].copy()
    else:
        df_owl_80830 = pd.DataFrame()
        df_owl_80831 = pd.DataFrame()

    all_dfs = []
    for s in sheet_names:
        if s == "80830":
            continue
        temp = pd.read_excel(xls, sheet_name=s)
        temp["motusTagID_sheet"] = pd.to_numeric(s, errors="coerce")
        all_dfs.append(temp)

    if len(df_owl_80830) > 0: all_dfs.append(df_owl_80830)
    if len(df_owl_80831) > 0: all_dfs.append(df_owl_80831)

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # standardize motusTagID (fallback to sheet label)
    if "motusTagID" not in df_combined.columns:
        df_combined["motusTagID"] = df_combined["motusTagID_sheet"]
    df_combined["motusTagID"] = pd.to_numeric(df_combined["motusTagID"], errors="coerce")

    return df_combined


@st.cache_data
def load_meta_csv(uploaded_csv):
    if uploaded_csv is None:
        return None
    meta = pd.read_csv(uploaded_csv, low_memory=False)

    # standardize ID col
    if "motusTagID" not in meta.columns and "tag_id" in meta.columns:
        meta = meta.rename(columns={"tag_id": "motusTagID"})

    meta["motusTagID"] = pd.to_numeric(meta["motusTagID"], errors="coerce")
    meta_one = meta.groupby("motusTagID").first().reset_index()
    return meta_one


@st.cache_data
def feature_engineer(df_combined, meta_one=None):
    # DATETIME, hour, TimeOfDay
    df_combined = build_datetime(df_combined)

    # Outlier capping
    numeric_cols = ["snr", "sig", "noise", "freq"]
    df_capped = cap_outliers_iqr(df_combined, numeric_cols)

    # Correlation drop @ 0.85
    num_cols = ["snr", "sig", "sigsd", "noise", "freq", "freqsd", "burstSlop", "slop"]
    num_cols = [c for c in num_cols if c in df_capped.columns]

    to_drop = []
    if len(num_cols) > 1:
        corr_matrix = df_capped[num_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > 0.85)]
        df_final = df_capped.drop(columns=to_drop)
    else:
        df_final = df_capped.copy()

    # merge metadata per motusTagID
    combined_final = df_final
    if meta_one is not None:
        combined_final = df_final.merge(meta_one, on="motusTagID", how="left")

    return df_combined, df_capped, df_final, combined_final, to_drop


@st.cache_data
def build_owl_level(combined_final):
    # True stay duration
    first_det = combined_final.groupby("motusTagID")["DATETIME"].min()
    last_det  = combined_final.groupby("motusTagID")["DATETIME"].max()
    stay_true = (last_det - first_det).dt.total_seconds() / (3600 * 24)
    stay_true = stay_true.clip(lower=0).reset_index()
    stay_true.columns = ["motusTagID", "stay_duration_days"]

    # Owl-level aggregates
    num_cols = ["snr","sigsd","freq","freqsd","slop","burstSlop","antBearing","port","nodeNum","runLen","hour"]
    num_cols = [c for c in num_cols if c in combined_final.columns]

    agg_dict = {"detections_count": ("motusTagID", "size")}
    for c in num_cols:
        agg_dict[f"{c}_mean"] = (c, safe_mean)
        agg_dict[f"{c}_std"]  = (c, safe_std)

    owl_features = combined_final.groupby("motusTagID").agg(**agg_dict).reset_index()

    owl_df = owl_features.merge(stay_true, on="motusTagID", how="left")

    # Residency bins
    bins = [0, 3, 7, np.inf]
    labels = ["Vagrant", "Migrant", "Resident"]
    owl_df["ResidencyType_true"] = pd.cut(
        owl_df["stay_duration_days"],
        bins=bins, labels=labels, include_lowest=True
    )

    return owl_df


def run_models(owl_df):
    # ---------------- REGRESSION ----------------
    y_reg = owl_df["stay_duration_days"]
    drop_cols = ["motusTagID","stay_duration_days","ResidencyType_true"]
    X = owl_df.drop(columns=[c for c in drop_cols if c in owl_df.columns], errors="ignore")

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

    rows, pred_store = [], {}
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows.append({
            "Model": name,
            "R2": r2_score(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred))
        })
        pred_store[name] = pred

    reg_results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    best_reg_name = reg_results.iloc[0]["Model"]
    best_reg = reg_models[best_reg_name]
    best_pred = pred_store[best_reg_name]

    # Fit best on full data + predict all owls
    best_reg.fit(X_imp, y_reg)
    owl_df["predicted_stay_days"] = best_reg.predict(X_imp)

    # ---------------- CLASSIFICATION ----------------
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
            "params": {"clf__C": np.logspace(-2,2,12), "clf__solver":["lbfgs","liblinear"]}
        },
        "SVC": {
            "estimator": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", SVC(class_weight="balanced"))
            ]),
            "params": {"clf__C": np.logspace(-2,2,12), "clf__gamma":["scale","auto"]}
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
            "params": {
                "n_estimators":[200,400,800],
                "max_depth":[None,6,12,20],
                "min_samples_leaf":[1,2,4,6]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cls_rows, best_models = [], {}

    for name, cfg in candidates.items():
        search = RandomizedSearchCV(
            cfg["estimator"], cfg["params"],
            n_iter=20, scoring="f1_macro",
            cv=cv, random_state=42, n_jobs=-1
        )
        search.fit(Xc_train, yc_train)

        best_models[name] = search.best_estimator_
        pred = best_models[name].predict(Xc_test)

        cls_rows.append({
            "Model": name,
            "BestCV_F1": search.best_score_,
            "Test_F1": f1_score(yc_test, pred, average="macro"),
            "Test_Acc": accuracy_score(yc_test, pred),
            "BestParams": search.best_params_
        })

    cls_results = pd.DataFrame(cls_rows).sort_values("Test_F1", ascending=False)
    best_cls_name = cls_results.iloc[0]["Model"]
    best_cls = best_models[best_cls_name]
    yc_pred = best_cls.predict(Xc_test)

    all_labels = np.arange(len(le.classes_))
    report = classification_report(
        yc_test, yc_pred,
        labels=all_labels,
        target_names=le.classes_,
        zero_division=0
    )
    cm = confusion_matrix(yc_test, yc_pred, labels=all_labels)

    return owl_df, reg_results, best_reg_name, y_test, best_pred, cls_results, best_cls_name, report, cm, le.classes_


# =====================================================
# AUTO RUN PIPELINE
# =====================================================
df_combined = load_detections_excel(xlsx_up)
meta_one = load_meta_csv(meta_up)

df_combined, df_capped, df_final, combined_final, to_drop = feature_engineer(df_combined, meta_one)
owl_df = build_owl_level(combined_final)

owl_df, reg_results, best_reg_name, y_test, best_pred, cls_results, best_cls_name, report, cm, class_names = run_models(owl_df)


# =====================================================
# TABS UI
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA (detections)",
    "🛠️ Feature Engineering",
    "🤖 Modeling",
    "📌 Final Results"
])

with tab1:
    st.header("Detections Dataset (combined from Excel sheets)")
    st.dataframe(df_combined.head(30))
    st.write("Shape:", df_combined.shape)

    st.subheader("Hourly detections")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df_combined["hour"], bins=24, ax=ax)
    ax.set_title("Hourly Detection Frequency")
    ax.set_xlabel("Hour")
    st.pyplot(fig)

    st.subheader("Night vs Day (%)")
    night_day = df_combined["TimeOfDay"].value_counts(normalize=True)*100
    fig, ax = plt.subplots(figsize=(6,4))
    night_day.plot(kind="bar", ax=ax)
    ax.set_ylabel("Percentage %")
    ax.set_title("Night vs Day Detection Percentage")
    st.pyplot(fig)

with tab2:
    st.header("Feature Engineering Outputs")

    st.subheader("Outlier capped detections preview")
    st.dataframe(df_capped.head(15))

    st.subheader("Dropped correlated columns (threshold=0.85)")
    st.write(to_drop)

    st.subheader("Final cleaned detections shape")
    st.write(df_final.shape)

    st.subheader("Combined detections + metadata preview")
    st.dataframe(combined_final.head(15))

    st.subheader("Owl-level dataset (engineered)")
    st.dataframe(owl_df.head(25))

with tab3:
    st.header("Modeling Results")

    st.subheader("Regression model comparison")
    st.dataframe(reg_results)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(y_test, best_pred, alpha=0.7)
    ax.set_xlabel("True stay duration (days)")
    ax.set_ylabel("Predicted stay duration (days)")
    ax.set_title(f"Best Regression Model: {best_reg_name}")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Classification model comparison")
    st.dataframe(cls_results)

    st.text("Classification Report:")
    st.text(report)

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {best_cls_name}")
    st.pyplot(fig)

with tab4:
    st.header("Final Summary (simple view)")

    st.write(f"Total owls analyzed: **{owl_df['motusTagID'].nunique()}**")
    st.write(f"Best regression model: **{best_reg_name}**")
    st.write(f"Best classification model: **{best_cls_name}**")

    st.subheader("Residency distribution")
    st.bar_chart(owl_df["ResidencyType_true"].value_counts())

    st.markdown("---")
    st.subheader("Download owl-level final dataset (safe, no raw detections)")
    st.download_button(
        "Download final_true_duration_dataset.csv",
        owl_df.to_csv(index=False).encode("utf-8"),
        file_name="final_true_duration_dataset.csv",
        mime="text/csv"
    )
