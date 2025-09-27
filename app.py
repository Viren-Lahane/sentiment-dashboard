# app.py
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
from io import BytesIO
import base64
import re

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# -----------------------
# Utilities
# -----------------------
def clean_text(s: str) -> str:
    """Basic cleaning: remove urls and extra whitespace."""
    s = re.sub(r"http\S+|www\S+|https\S+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\@\w+|\#", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_pipeline():
    """Cache pipeline in session state to avoid reloading."""
    if "nlp" not in st.session_state:
        st.session_state["nlp"] = pipeline("sentiment-analysis")
    return st.session_state["nlp"]

def analyze_texts(nlp, texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = nlp(batch)
        results.extend(preds)
    return results

def df_from_preds(texts, preds):
    df = pd.DataFrame({
        "text": texts,
        "label": [p["label"] for p in preds],
        "score": [p["score"] for p in preds]
    })
    return df

def make_wordcloud(texts):
    if not texts:
        return None
    text_blob = " ".join(texts)
    wc = WordCloud(width=800, height=400, collocations=False).generate(text_blob)
    return wc.to_image()

def to_download_link(df, filename="predictions.csv"):
    return df.to_csv(index=False).encode('utf-8')

# -----------------------
# UI Layout
# -----------------------
st.title("✨ Sentiment Analysis Dashboard")
st.markdown("Upload a CSV of texts (or paste a sentence) and get sentiment labels & visualizations.")

nlp = get_pipeline()

# Sidebar options
with st.sidebar:
    st.header("Options")
    batch_size = st.number_input("Batch size (inference)", min_value=1, max_value=256, value=32)
    show_wordcloud = st.checkbox("Show wordclouds", value=True)
    st.markdown("---")
    st.markdown("**Notes:** The app uses a pretrained transformer sentiment pipeline.")

# Main - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single text analysis")
    text_input = st.text_area("Enter text here", height=120, value="")
    if st.button("Analyze text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(text_input)
            with st.spinner("Analyzing..."):
                pred = nlp(cleaned[:1000])  # single-call
            st.metric("Label", pred[0]["label"])
            st.write(f"Confidence: {pred[0]['score']:.3f}")
            st.write("**Cleaned text:**")
            st.write(cleaned)

with col2:
    st.subheader("Batch upload (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file with a text column", type=["csv"])
    sample_button = st.button("Use sample data")
    df = None
    if sample_button:
        # Simple sample
        df = pd.DataFrame({
            "text": [
                "I love this product! It's amazing.",
                "Worst experience ever. Will not buy again.",
                "I had an okay day, nothing special."
            ]
        })

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    if df is not None:
        # detect text columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            st.error("No text-like column found in the uploaded CSV.")
        else:
            default_col = "text" if "text" in text_cols else text_cols[0]
            chosen_col = st.selectbox("Select text column", text_cols, index=text_cols.index(default_col))
            texts = df[chosen_col].astype(str).apply(clean_text).tolist()

            if st.button("Run batch sentiment"):
                with st.spinner("Running sentiment analysis..."):
                    preds = analyze_texts(nlp, texts, batch_size=batch_size)
                    out_df = df_from_preds(texts, preds)
                    st.success("Done!")
                    st.dataframe(out_df.head(50))

                    # Visuals
                    st.subheader("Label distribution")
                    counts = out_df['label'].value_counts().reset_index()
                    counts.columns = ['label','count']
                    fig = px.bar(counts, x='label', y='count', title='Sentiment distribution')
                    st.plotly_chart(fig, use_container_width=True)

                    # Wordclouds
                    if show_wordcloud:
                        pos_texts = out_df[out_df['label']=='POSITIVE']['text'].tolist()
                        neg_texts = out_df[out_df['label']=='NEGATIVE']['text'].tolist()
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.write("**Positive wordcloud**")
                            img = make_wordcloud(pos_texts)
                            if img:
                                st.image(img, use_column_width=True)
                            else:
                                st.write("No positive texts.")
                        with col_neg:
                            st.write("**Negative wordcloud**")
                            img2 = make_wordcloud(neg_texts)
                            if img2:
                                st.image(img2, use_column_width=True)
                            else:
                                st.write("No negative texts.")

                    # Download
                    csv_bytes = to_download_link(out_df)
                    st.download_button("Download predictions as CSV", csv_bytes, file_name="predictions.csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️  — use this app to analyze small to medium sized datasets. For very large data, run batched offline jobs.")
