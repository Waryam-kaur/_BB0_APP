import streamlit as st
import pandas as pd

# ================== CONFIG ==================
# TODO: change this to the actual path of your data file
CSV_PATH = "your_data.csv"  # e.g. "data/cleaned_data.csv"


# ================== DATA LOADING ==================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ================== SIMPLE CHATBOT LOGIC ==================
def answer_question(question: str, df: pd.DataFrame) -> str:
    """
    Very lightweight, rule-based 'chatbot' that uses the dataset.
    It checks the text of the question and returns a numeric/stat answer.
    """
    q = question.lower()

    # ---- Basic info about dataset ----
    if "how many rows" in q or "how many records" in q or "how many observations" in q:
        return f"The dataset has {len(df)} rows (records)."

    if "how many columns" in q or "how many features" in q:
        return f"The dataset has {df.shape[1]} columns. They are: {', '.join(df.columns)}."

    if "columns" in q or "features" in q and "list" in q:
        return f"The columns in the dataset are: {', '.join(df.columns)}."

    # ---- Helper: map keywords to column names ----
    def detect_column(q_text: str) -> str | None:
        # Adjust these mappings based on your actual column names
        mapping = {
            "stay": "stay_duration",
            "duration": "stay_duration",
            "age": "age",
            "cost": "cost",
            "charge": "total_charges",
            "length of stay": "stay_duration",
        }
        for key, col in mapping.items():
            if key in q_text and col in df.columns:
                return col
        return None

    col = detect_column(q)

    # ---- Stats on a single numeric column ----
    if col:
        if "average" in q or "mean" in q:
            return f"The average of '{col}' is {df[col].mean():.2f}."

        if "minimum" in q or "min" in q or "lowest" in q:
            return f"The minimum value of '{col}' is {df[col].min():.2f}."

        if "maximum" in q or "max" in q or "highest" in q:
            return f"The maximum value of '{col}' is {df[col].max():.2f}."

        if "median" in q:
            return f"The median of '{col}' is {df[col].median():.2f}."

        if "std" in q or "standard deviation" in q:
            return f"The standard deviation of '{col}' is {df[col].std():.2f}."

        if "describe" in q or "summary" in q:
            desc = df[col].describe().to_dict()
            pretty = ", ".join([f"{k}: {v:.2f}" for k, v in desc.items() if isinstance(v, (int, float, float))])
            return f"Summary stats for '{col}' → {pretty}"

    # ---- Residency / stay_duration explanation (your project logic) ----
    if "residency" in q or "resident" in q:
        return (
            "The original dataset does not contain a 'residency_type' column. "
            "We created residency_type using a machine learning model that "
            "predicts it from 'stay_duration'."
        )

    if "stay duration" in q or "stay" in q and "used" in q:
        return (
            "Stay duration is the main feature used as input to the model. "
            "The model learns the relationship between stay_duration and the "
            "final residency type."
        )

    if "model" in q or "machine learning" in q or "ml" in q:
        return (
            "The machine learning model was trained offline using this dataset. "
            "In this chatbot app we are only exploring the data with simple "
            "statistics, not training the model again."
        )

    if "dataset" in q or "data" in q:
        return (
            "This dataset is used to understand patterns such as how stay_duration "
            "behaves. Residency_type itself is not a raw column, it is generated "
            "by an ML model from stay_duration."
        )

    # ---- Fallback ----
    return (
        "I can help you with basic questions about the dataset.\n\n"
        "Try asking things like:\n"
        "- 'How many rows are in the dataset?'\n"
        "- 'How many columns does the dataset have?'\n"
        "- 'What is the average stay duration?'\n"
        "- 'What are the columns in the dataset?'\n"
        "- 'How is residency_type related to stay_duration?'"
    )


# ================== STREAMLIT APP ==================
def main():
    st.title("📊 Data Chatbot for Residency Project")

    st.markdown(
        "This is a simple chatbot that answers questions about the dataset used "
        "in the residency prediction project. It uses the **same data**, but does "
        "not retrain any models."
    )

    # Load data
    try:
        df = load_data(CSV_PATH)
    except FileNotFoundError:
        st.error(
            f"Could not find the data file at `{CSV_PATH}`. "
            "Please update CSV_PATH at the top of `chatbot_app.py`."
        )
        return

    # Show a quick preview in an expander
    with st.expander("🔍 Show data preview"):
        st.write(df.head())

    # Chat interface
    st.subheader("💬 Ask the chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Type your question about the data:")

    if st.button("Ask"):
        if user_q.strip():
            answer = answer_question(user_q, df)
            st.session_state.chat_history.append(("You", user_q))
            st.session_state.chat_history.append(("Bot", answer))

    # Display chat history
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")


if __name__ == "__main__":
    main()
