import streamlit as st
import pandas as pd

# ================== AVAILABLE DATA FILES ==================
DATA_FILES = {
    "Selected Dataset (Used for ML Model)": "clean_df_selected.csv",
    "Combined Dataset (Full Saw-whet Owl Data)": "combined_sawwhet_owls.csv",
}

# ================== LOAD DATA ==================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ================== CHATBOT LOGIC ==================
def answer_question(question: str, df: pd.DataFrame) -> str:
    q = question.lower()

    # Basic dataset-level questions
    if "how many rows" in q or "how many records" in q:
        return f"The dataset has **{len(df)} rows**."

    if "how many columns" in q or "how many features" in q:
        return (
            f"The dataset has **{df.shape[1]} columns**.\n\n"
            f"Columns: {', '.join(df.columns)}"
        )

    if "columns" in q and ("list" in q or "names" in q):
        return f"The columns are:\n\n{', '.join(df.columns)}"

    # Detect column mentioned by user
    def detect_column(q_text: str):
        for col in df.columns:
            if col.lower() in q_text:
                return col
        return None

    col = detect_column(q)

    # Stats for numeric columns
    if col and pd.api.types.is_numeric_dtype(df[col]):
        if "average" in q or "mean" in q:
            return f"The average of **{col}** is **{df[col].mean():.2f}**."

        if "min" in q or "minimum" in q:
            return f"The minimum of **{col}** is **{df[col].min():.2f}**."

        if "max" in q or "maximum" in q:
            return f"The maximum of **{col}** is **{df[col].max():.2f}**."

        if "median" in q:
            return f"The median of **{col}** is **{df[col].median():.2f}**."

        if "std" in q or "standard deviation" in q:
            return f"The standard deviation of **{col}** is **{df[col].std():.2f}**."

        if "describe" in q or "summary" in q:
            return df[col].describe().to_frame().to_markdown()

    # Residency logic from ML project
    if "residency" in q or "resident" in q:
        return (
            "The dataset does NOT originally contain a `residency_type` column.\n"
            "We **created residency_type using an ML model** that predicts it based on stay_duration."
        )

    if "stay duration" in q or "stay" in q:
        return (
            "`stay_duration` is the most important feature and is used by the ML model "
            "to predict residency_type."
        )

    if "model" in q or "machine learning" in q or "ml" in q:
        return (
            "The ML model was trained on the SELECTED dataset (`clean_df_selected.csv`).\n"
            "The chatbot only explores the data, it does not retrain the model."
        )

    if "dataset" in q or "data" in q:
        return (
            "The dataset helps us understand patterns like stay duration, "
            "timings, counts, and owl characteristics used in the project."
        )

    # Default fallback
    return (
        "I can help answer questions about the dataset.\n\n"
        "Try asking:\n"
        "- How many rows are there?\n"
        "- What columns are in the dataset?\n"
        "- What is the mean stay_duration?\n"
        "- Describe age or wing_length.\n"
        "- How is residency_type created?\n"
    )


# ================== STREAMLIT APP ==================
def main():
    st.title("🦉 Owl Data Chatbot (Using Your Project Datasets)")

    st.markdown(
        "This chatbot can answer questions about the two datasets used in your project:\n"
        "- **clean_df_selected.csv** → ML-ready filtered dataset\n"
        "- **combined_sawwhet_owls.csv** → Full combined owl dataset"
    )

    # Select dataset
    dataset_choice = st.selectbox("Choose a dataset:", list(DATA_FILES.keys()))
    csv_path = DATA_FILES[dataset_choice]

    # Load dataset
    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        st.error(f"❌ File not found: `{csv_path}`")
        return

    st.success(f"📁 Loaded **{dataset_choice}** from `{csv_path}`")

    # Preview
    with st.expander("🔍 Show Data Preview"):
        st.dataframe(df.head())

    # Chat section
    st.subheader("💬 Ask the Chatbot About the Data")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Type your question:")

    if st.button("Ask"):
        if user_q.strip():
            reply = answer_question(user_q, df)
            st.session_state.chat_history.append(("You", user_q))
            st.session_state.chat_history.append(("Bot", reply))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")


if __name__ == "__main__":
    main()
