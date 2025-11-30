import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.inspection import permutation_importance


warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global residency thresholds
# -------------------------------------------------------------------
SHORT_THR = 3.0   # 0–3 days -> Vagrant
LONG_THR = 7.0    # 3–7 days -> Migrant ; 7+ -> Resident

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(page_title="BBO Owl Migration App", layout="wide")
st.title("🦉 BBO Owl Migration App (EDA + FE + Modeling + XAI + RAG Chatbot)")

# -------------------------------------------------------------------
# Helper EDA functions
# -------------------------------------------------------------------
def plot_detections_per_owl(df_det):
    st.subheader("1. Detections per Owl (motusTagID)")
    if "motusTagID" not in df_det.columns:
        st.warning("`motusTagID` column not found in detection data.")
        return

    counts = df_det["motusTagID"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Detections per Owl (motusTagID)")
    ax.set_xlabel("motusTagID")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def cap_outliers_iqr(df, col):
    """IQR-based capping exactly like your notebook."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_cap = Q1 - 1.5 * IQR
    upper_cap = Q3 + 1.5 * IQR
    return df[col].clip(lower=lower_cap, upper=upper_cap)


def plot_before_after_boxplots(df_det):
    st.subheader("2. Outlier Capping (Before vs After)")

    signal_cols = ["snr", "sig", "sigsd", "noise", "freq", "freqsd", "burstSlop", "slop"]
    cols_present = [c for c in signal_cols if c in df_det.columns]

    if not cols_present:
        st.warning("No numeric signal columns found for outlier capping plots.")
        return

    if "snr" in cols_present and "sig" in cols_present:
        cols_to_show = ["snr", "sig"]
    else:
        cols_to_show = cols_present[:2]

    df_capped = df_det.copy()
    for c in cols_present:
        df_capped[c] = cap_outliers_iqr(df_capped, c)

    for col in cols_to_show:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(x=df_det[col], ax=axes[0])
        axes[0].set_title(f"Before Capping: {col}")
        sns.boxplot(x=df_capped[col], ax=axes[1])
        axes[1].set_title(f"After Capping: {col}")
        plt.tight_layout()
        st.pyplot(fig)


def plot_signal_corr_heatmap(df_det):
    st.subheader("3. Correlation Heatmap (Numeric Signal Features)")

    signal_cols = ["snr", "sig", "sigsd", "noise", "freq", "freqsd", "burstSlop", "slop"]
    cols_present = [c for c in signal_cols if c in df_det.columns]

    if len(cols_present) < 2:
        st.warning("Not enough numeric signal features for a correlation heatmap.")
        return

    corr = df_det[cols_present].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Signal Features)")
    st.pyplot(fig)


def eda_summary_text():
    st.subheader("4. EDA & Feature Engineering Summary (From Notebook)")
    st.markdown(
        """
We started with a multi-sheet MOTUS dataset containing raw owl detections from 42 receiver stations.  
Each sheet corresponded to one Northern Saw-whet Owl, except sheet **80830**, which contained detections
from **two** different owls (80830 and 80831). We identified this by checking the unique `motusTagID`
values inside each sheet and then manually split that sheet into two clean subsets.

All sheets were concatenated into a single detection-level dataframe. Pandas automatically aligned
shared columns and filled missing columns with `NaN` where a sheet did not have a field.

We cleaned the timestamp fields by converting `DATE` and `TIME` into a unified **`DATETIME`** column.
From this, we later derived features such as **hour of day** and daily detection counts.  
Next, we handled structural missing values and performed **IQR-based capping (winsorizing)** for
signal-based numeric columns (e.g., `snr`, `sig`, `noise`) to reduce extreme spikes without destroying
biological variation.

Finally, we used a correlation heatmap to drop highly correlated features and merged the cleaned
new MOTUS detections with the old metadata file (age, sex, species, measurements) via `motusTagID`.
This produced a fully enriched dataset that was ready for modelling.
        """
    )


# -------------------------------------------------------------------
# Main Modelling / FE helper
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded):
    if uploaded is None:
        return None
    return pd.read_csv(uploaded, low_memory=False)

@st.cache_data(show_spinner=True)
def prepare_model_data(df_new, df_old):
    """Feature engineering + LIGHT modelling (no RandomizedSearchCV)."""

    df = df_new.copy()

    # 1) Standardize motusTagID column
    possible_id_cols = [c for c in df.columns if "motus" in c.lower() and "tag" in c.lower()]
    possible_id_cols += [c for c in df.columns if c.lower() in ["tag_id", "tagid", "motustagid_sheet"]]
    if not possible_id_cols:
        raise ValueError("Could not find a motus tag ID column in the NEW detections file.")
    id_col = possible_id_cols[0]
    if id_col != "motusTagID":
        df.rename(columns={id_col: "motusTagID"}, inplace=True)

    df["motusTagID"] = pd.to_numeric(df["motusTagID"], errors="coerce")

    # 2) Build DATETIME
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

    # hour-of-day feature
    df["hour"] = df["DATETIME"].dt.hour

    # 3) True stay duration (days) per owl
    first_det = df.groupby("motusTagID")["DATETIME"].min()
    last_det = df.groupby("motusTagID")["DATETIME"].max()

    stay_true = (last_det - first_det).dt.total_seconds() / (3600 * 24)
    stay_true = stay_true.clip(lower=0)
    stay_true = stay_true.reset_index()
    stay_true.columns = ["motusTagID", "stay_duration_days"]

    # 4) Aggregated owl features
    def safe_mean(s):
        return pd.to_numeric(s, errors="coerce").mean()

    def safe_std(s):
        return pd.to_numeric(s, errors="coerce").std()

    numeric_candidates = [
        "snr",
        "sig",
        "sigsd",
        "noise",
        "freq",
        "freqsd",
        "slop",
        "burstSlop",
        "antBearing",
        "port",
        "nodeNum",
        "runLen",
        "hour",
    ]
    num_cols = [c for c in numeric_candidates if c in df.columns]

    agg_dict = {"detections_count": ("motusTagID", "size")}
    for c in num_cols:
        agg_dict[f"{c}_mean"] = (c, safe_mean)
        agg_dict[f"{c}_std"] = (c, safe_std)

    owl_features = df.groupby("motusTagID").agg(**agg_dict).reset_index()

    # 5) Old metadata (one row per owl)
    df_old_local = df_old.copy()
    old_id_candidates = [
        c
        for c in df_old_local.columns
        if ("motus" in c.lower() and "tag" in c.lower()) or c.lower() in ["tag_id", "tagid", "motustagid_sheet"]
    ]
    if not old_id_candidates:
        raise ValueError("Could not find a motus tag ID column in the OLD metadata file.")
    old_id_col = old_id_candidates[0]

    if old_id_col != "motusTagID":
        df_old_local.rename(columns={old_id_col: "motusTagID"}, inplace=True)

    df_old_local["motusTagID"] = pd.to_numeric(df_old_local["motusTagID"], errors="coerce")

    # ONE metadata row per owl (first)
    meta_one = df_old_local.groupby("motusTagID").first().reset_index()

    # 6) Merge features + stay duration + metadata
    owl_df = owl_features.merge(stay_true, on="motusTagID", how="left")
    owl_df = owl_df.merge(meta_one, on="motusTagID", how="left", suffixes=("", "_meta"))

    # 7) Residency type from thresholds
    bins = [0, SHORT_THR, LONG_THR, np.inf]
    labels = ["Vagrant", "Migrant", "Resident"]
    owl_df["ResidencyType_true"] = pd.cut(
        owl_df["stay_duration_days"], bins=bins, labels=labels, include_lowest=True
    )

    # ---------------- REGRESSION ----------------
    y_reg = owl_df["stay_duration_days"]
    drop_cols = ["motusTagID", "stay_duration_days", "ResidencyType_true"]
    X_reg = owl_df.drop(columns=[c for c in drop_cols if c in owl_df.columns])

    X_reg = X_reg.select_dtypes(include=[np.number]).copy()
    X_reg = X_reg.dropna(axis=1, how="all")

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_reg), columns=X_reg.columns, index=X_reg.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y_reg, test_size=0.2, random_state=42
    )

    reg_models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.2)),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    reg_rows = []
    pred_store_reg = {}

    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        reg_rows.append({"Model": name, "R2": r2, "RMSE": rmse})
        pred_store_reg[name] = pred

    reg_results = pd.DataFrame(reg_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    best_reg_name = reg_results.iloc[0]["Model"]
    best_reg = reg_models[best_reg_name]

    # Train on full data and store predictions
    best_reg.fit(X_imp, y_reg)
    best_pred_full = best_reg.predict(X_imp)
    owl_df["predicted_stay_days"] = best_pred_full

    # ---------------- CLASSIFICATION ----------------
    y_cls = owl_df["ResidencyType_true"].astype(str)
    le = LabelEncoder()
    y_cls_enc = le.fit_transform(y_cls)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_imp, y_cls_enc, test_size=0.2, random_state=42, stratify=y_cls_enc
    )

    cls_models = {
        "LogReg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", SVC(class_weight="balanced", C=1.0, gamma="scale")),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced_subsample", random_state=42, n_estimators=200, max_depth=None
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42, n_estimators=150, learning_rate=0.1, max_depth=3
        ),
    }

    cls_rows = {}
    best_cls = None
    best_cls_name = None
    best_f1 = -1.0

    for name, model in cls_models.items():
        model.fit(Xc_train, yc_train)
        preds = model.predict(Xc_test)
        f1 = f1_score(yc_test, preds, average="macro")
        cls_rows[name] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_cls = model
            best_cls_name = name

    cls_results = (
        pd.DataFrame(
            [{"Model": name, "F1_macro": score} for name, score in cls_rows.items()]
        )
        .sort_values("F1_macro", ascending=False)
        .reset_index(drop=True)
    )

    yc_pred = best_cls.predict(Xc_test)

    return {
        "df_det": df,
        "owl_df": owl_df,
        "X_imp": X_imp,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "reg_results": reg_results,
        "best_reg_name": best_reg_name,
        "best_reg": best_reg,
        "best_reg_pred_test": pred_store_reg[best_reg_name],
        "Xc_train": Xc_train,
        "Xc_test": Xc_test,
        "yc_train": yc_train,
        "yc_test": yc_test,
        "le": le,
        "cls_results": cls_results,
        "best_cls_name": best_cls_name,
        "best_cls": best_cls,
        "yc_pred": yc_pred,
    }

# -------------------------------------------------------------------
# RAG Chatbot (based on RAG_Implementation.ipynb sample notebook)
# -------------------------------------------------------------------
# In the professor's Colab:
#   - They create small "documents" (charity_info + transaction_narrative)
#   - They implement retrieve_context(query, top_k) using embeddings
#   - They implement rag_chatbot(query) that calls retrieve_context + LLM
#
# Here we follow the *same structure*, but use a lightweight
# keyword-based retrieval so it runs safely on Streamlit Cloud.

@st.cache_data
def build_owl_documents(owl_df: pd.DataFrame):
    """
    STEP 1: BUILD DOCUMENTS  
    (analogous to 'documents' dict in RAG_Implementation.ipynb)

    In the Colab example, the professor builds two documents:
        - doc1: charity_info
        - doc2: transaction_narrative (from a DataFrame)

    Here we do the same idea, but each document is a short narrative
    about ONE owl's stay, detections, and residency type.
    """
    docs = []
    for _, row in owl_df.iterrows():
        if pd.isna(row.get("stay_duration_days")):
            continue

        motus_id = row.get("motusTagID")
        stay = row.get("stay_duration_days")
        residency = row.get("ResidencyType_true")
        detections = row.get("detections_count")

        text = (
            f"Owl {int(motus_id)} stayed at the station for about "
            f"{stay:.1f} days, with {int(detections)} detections, "
            f"and was categorized as a {residency}."
        )
        docs.append(text)

    return docs


def simple_retrieve_context(query: str, docs, top_k: int = 5) -> str:
    """
    STEP 2: RETRIEVE CONTEXT  
    (same role as retrieve_context(query, top_k) in the Colab)

    Professor's notebook:
        - encodes docs with SentenceTransformer
        - uses cosine similarity to find the most similar docs

    Here, to avoid heavy models on Streamlit Cloud, we implement a
    *lightweight* retrieval:
        - lowercase the query
        - count how many query words appear in each document
        - return the top_k matching documents as the context
    """
    if not docs or not query.strip():
        return ""

    q_words = [w for w in query.lower().split() if len(w) > 2]
    if not q_words:
        return ""

    scored = []
    for idx, text in enumerate(docs):
        t = text.lower()
        score = sum(1 for w in q_words if w in t)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return ""

    scored.sort(reverse=True)  # highest score first
    top_idx = [idx for score, idx in scored[:top_k]]
    top_texts = [docs[i] for i in top_idx]
    return "\n\n".join(top_texts)


def rag_chatbot_owl(query: str, docs):
    """
    STEP 3: RAG CHATBOT  
    (same role as rag_chatbot(query) in RAG_Implementation.ipynb)

    Pipeline:
        1. Retrieve context using simple_retrieve_context
        2. Generate a short natural-language answer based ONLY on that context
    """
    context = simple_retrieve_context(query, docs, top_k=5)

    if not context:
        answer = (
            "I couldn't find relevant information in the owl residency data "
            "for that question. Try asking about stay duration, detections, "
            "or residency types (Vagrant / Migrant / Resident)."
        )
    else:
        answer = (
            "Here is what I found in the owl residency data related to your question:\n\n"
            f"{context}\n\n"
            "This answer is based only on the detected stay duration, number of detections, "
            "and residency category for each owl."
        )

    return answer, context

# -------------------------------------------------------------------
# Sidebar – file upload
# -------------------------------------------------------------------
st.sidebar.header("1) Upload data")

st.sidebar.markdown("**Upload your final combined detection dataset and old metadata CSVs.**")

combined_file = st.sidebar.file_uploader(
    "FINAL combined dataset (combined_sawwhet_owls.csv)", type=["csv"], key="combined"
)

old_meta_file = st.sidebar.file_uploader(
    "OLD metadata (clean_df_selected.csv)", type=["csv"], key="oldmeta"
)

st.sidebar.markdown("---")
st.sidebar.write("Residency thresholds (fixed):")
st.sidebar.write(f"- 0–{SHORT_THR} days → **Vagrant**")
st.sidebar.write(f"- {SHORT_THR}–{LONG_THR} days → **Migrant**")
st.sidebar.write(f"- {LONG_THR}+ days → **Resident**")

# -------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------
if (combined_file is None) or (old_meta_file is None):
    st.info("Please upload **both** CSV files in the sidebar to see EDA, modelling, XAI, and the RAG chatbot.")
else:
    df_new = load_csv_from_upload(combined_file)
    df_old = load_csv_from_upload(old_meta_file)

    st.write("### Uploaded Data Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**New MOTUS detections (combined_sawwhet_owls.csv)**")
        st.write(df_new.head())
        st.write(f"Shape: {df_new.shape}")
    with c2:
        st.write("**Old metadata (clean_df_selected.csv)**")
        st.write(df_old.head())
        st.write(f"Shape: {df_old.shape}")

    # Prepare datasets & models
    data_dict = prepare_model_data(df_new, df_old)

    df_det = data_dict["df_det"]
    owl_df = data_dict["owl_df"]

   tab_eda, tab_model, tab_xai, tab_rag = st.tabs(
    ["📊 EDA & Feature Engineering",
     "🤖 Modelling & Results",
     "🔍 XAI",
     "💬 RAG Chatbot"]
)

    )

    # -------------------------------------------------------------------
    # TAB 1: EDA & Feature Engineering
    # -------------------------------------------------------------------
    with tab_eda:
        st.header("Exploratory Data Analysis (New MOTUS Detections)")

        st.subheader("Raw Detection-Level Snapshot")
        st.write(df_det.head())

        plot_detections_per_owl(df_det)
        plot_before_after_boxplots(df_det)
        plot_signal_corr_heatmap(df_det)
        eda_summary_text()

    # -------------------------------------------------------------------
    # TAB 2: Modelling & Results
    # -------------------------------------------------------------------
    with tab_model:
        st.header("Regression & Classification Modelling")

        st.subheader("1. Final Owl-Level Modelling Dataset")
        st.write(f"Final dataset shape: {owl_df.shape}")
        st.write(f"Final unique owls: {owl_df['motusTagID'].nunique()}")
        st.dataframe(owl_df.head())

        # Regression results
        st.subheader("2. Regression Models – Predicting Stay Duration (days)")
        st.write(data_dict["reg_results"])

        best_reg_name = data_dict["best_reg_name"]
        st.markdown(f"**Best regression model:** `{best_reg_name}`")

        # Scatter: true vs predicted (test)
        y_test = data_dict["y_test"]
        best_pred_test = data_dict["best_reg_pred_test"]

        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
        ax_scatter.scatter(y_test, best_pred_test, alpha=0.7, edgecolor="black")
        ax_scatter.set_xlabel("True Stay Duration (days)")
        ax_scatter.set_ylabel("Predicted Stay Duration (days)")
        ax_scatter.set_title(f"Regression: True vs Predicted Stay Duration\nModel = {best_reg_name}")
        ax_scatter.grid(True)
        st.pyplot(fig_scatter)

        # Residency distribution (true labels)
        st.subheader("3. Residency Type Distribution (True Labels)")
        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        owl_df["ResidencyType_true"].value_counts().plot(kind="bar", ax=ax_res)
        ax_res.set_title("True Residency Type Distribution (New Motus Data)")
        ax_res.set_xlabel("ResidencyType_true")
        ax_res.set_ylabel("Owls")
        st.pyplot(fig_res)

        # Classification report + confusion matrix
        st.subheader("4. Classification Performance (Residency Type)")
        best_cls_name = data_dict["best_cls_name"]
        best_cls = data_dict["best_cls"]
        yc_test = data_dict["yc_test"]
        yc_pred = data_dict["yc_pred"]
        le = data_dict["le"]

        all_labels = np.arange(len(le.classes_))

        st.markdown(f"**Best classification model:** `{best_cls_name}`")
        report_txt = classification_report(
            yc_test, yc_pred, labels=all_labels, target_names=le.classes_, zero_division=0
        )
        st.text(report_txt)

        cm = confusion_matrix(yc_test, yc_pred, labels=all_labels)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title(f"Confusion Matrix — {best_cls_name}")
        st.pyplot(fig_cm)

        # Extra modelling visualizations
        st.subheader("5. Top 15 Owls by Number of Days Stayed")
        top15 = owl_df.sort_values("stay_duration_days", ascending=False).head(15)
        fig_top, ax_top = plt.subplots(figsize=(10, 4))
        ax_top.bar(top15["motusTagID"].astype(str), top15["stay_duration_days"])
        ax_top.set_title("Top 15 Owls by Number of Days Stayed")
        ax_top.set_xlabel("Owl ID")
        ax_top.set_ylabel("Stay Duration (days)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_top)

        st.subheader("6. What Time of Day Owls Are Most Active")
        if "hour" in df_det.columns:
            fig_hr, ax_hr = plt.subplots(figsize=(9, 4))
            sns.histplot(df_det["hour"].dropna(), bins=24, kde=False, ax=ax_hr)
            ax_hr.set_title("What Time of Day Owls Are Most Active")
            ax_hr.set_xlabel("Hour of Day")
            ax_hr.set_ylabel("Detections")
            st.pyplot(fig_hr)
        else:
            st.warning("`hour` column missing from detection-level data; cannot plot hourly activity.")

        st.subheader("7. Daily Owl Activity at the Station")
        if "DATETIME" in df_det.columns:
            df_det["date_only"] = df_det["DATETIME"].dt.date
            daily = df_det.groupby("date_only")["motusTagID"].count().reset_index()
            fig_daily, ax_daily = plt.subplots(figsize=(12, 5))
            ax_daily.plot(daily["date_only"], daily["motusTagID"], marker="o")
            ax_daily.set_title("Daily Owl Activity at the Station")
            ax_daily.set_xlabel("Date")
            ax_daily.set_ylabel("Number of Detections")
            plt.xticks(rotation=45, ha="right")
            ax_daily.grid(True)
            st.pyplot(fig_daily)
        else:
            st.warning("`DATETIME` column missing from detection-level data; cannot plot daily activity.")

    # -------------------------------------------------------------------
    # TAB 3: XAI (Permutation Importance)
    # -------------------------------------------------------------------
    with tab_xai:
        st.header("Explainable AI (XAI)")

        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]
        Xc_test = data_dict["Xc_test"]
        yc_test = data_dict["yc_test"]

        st.subheader("1. Feature Importance for Regression (Permutation Importance)")
        best_reg = data_dict["best_reg"]

        perm_reg = permutation_importance(
            best_reg, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1
        )
        reg_imp = pd.DataFrame(
            {"feature": X_test.columns, "importance": perm_reg.importances_mean}
        ).sort_values("importance", ascending=False)

        st.write("Top 10 most important features driving **stay_duration_days** predictions:")
        st.write(reg_imp.head(10))

        fig_reg_imp, ax_reg_imp = plt.subplots(figsize=(8, 4))
        ax_reg_imp.barh(
            reg_imp["feature"].head(10)[::-1],
            reg_imp["importance"].head(10)[::-1],
        )
        ax_reg_imp.set_title("Permutation Feature Importance (Regression)")
        ax_reg_imp.set_xlabel("Mean Importance")
        st.pyplot(fig_reg_imp)

        st.subheader("2. Feature Importance for Classification (Permutation Importance)")
        best_cls = data_dict["best_cls"]

        perm_cls = permutation_importance(
            best_cls, Xc_test, yc_test, n_repeats=20, random_state=42, n_jobs=-1
        )
        cls_imp = pd.DataFrame(
            {"feature": Xc_test.columns, "importance": perm_cls.importances_mean}
        ).sort_values("importance", ascending=False)

        st.write("Top 10 most important features for **ResidencyType_true** classification:")
        st.write(cls_imp.head(10))

        fig_cls_imp, ax_cls_imp = plt.subplots(figsize=(8, 4))
        ax_cls_imp.barh(
            cls_imp["feature"].head(10)[::-1],
            cls_imp["importance"].head(10)[::-1],
        )
        ax_cls_imp.set_title("Permutation Feature Importance (Classification)")
        ax_cls_imp.set_xlabel("Mean Importance")
        st.pyplot(fig_cls_imp)

        st.markdown(
            """
These importance plots help BBO staff understand **which behavioural or tag-related
features most strongly influence**:

- How long an owl stays near the station (regression target), and  
- Whether an owl behaves like a Vagrant, Migrant, or Resident (classification target).

This makes the model decisions more **transparent and actionable** for future research.
            """
        )
      # -------------------------------------------------------------------
# TAB 4: RAG Chatbot (follows RAG_Implementation.ipynb structure)
# -------------------------------------------------------------------
with tab_rag:
    st.header("💬 RAG Chatbot – Ask Questions About Owl Residency")

    st.markdown(
        """
This chatbot follows the same **RAG pattern** as in the course Colab
notebook `RAG_Implementation.ipynb`:

1. **Build documents** from data  
   - In the notebook: `charity_info` and `transaction_narrative`  
   - Here: one short narrative per owl (stay days, detections, residency).

2. **Retrieve context** for your question  
   - In the notebook: `retrieve_context(query, top_k)` with embeddings.  
   - Here: `simple_retrieve_context(query, docs, top_k)` using a light
     keyword-based similarity (safer for Streamlit Cloud limits).

3. **Generate an answer** based only on the retrieved context  
   - In the notebook: `rag_chatbot(query)` with FLAN-T5.  
   - Here: `rag_chatbot_owl(query, docs)` which creates a concise,
     rule-based answer.

You can use this chatbot to explore owl stay duration, residency types
(Vagrant / Migrant / Resident), and detection counts.
        """
    )

    # Build knowledge base (documents) once from the owl-level dataset
    docs = build_owl_documents(owl_df)

    user_q = st.text_input(
        "Type your question about owl stay duration, residency type, or detections:",
        placeholder="e.g., Which owls stayed the longest, or what is a Resident owl?"
    )

    if st.button("Ask the RAG Chatbot"):
        if not user_q.strip():
            st.warning("Please enter a question first.")
        elif not docs:
            st.warning("No owl documents available yet.")
        else:
            answer, used_context = rag_chatbot_owl(user_q, docs)

            st.subheader("Chatbot Answer")
            st.write(answer)

            with st.expander("Show retrieved context from owl data"):
                st.markdown(f"```text\n{used_context}\n```")
