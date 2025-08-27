import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

model_path = "assets/sentiment_lstm.keras"
tokenizer_path = "assets/tokenizer.json"
maxlen=200


def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None,None

def predict_sentiment(review_text, model, tokenizer):
    if not review_text.strip():
        return None, None
    
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence)

    prediction = model.predict(padded_sequence)
    sentiment_score = prediction[0][0]

    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"

    return sentiment , sentiment_score

st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown(
    "Enter a movie review below to find out whether the sentiment is **Positive** or **Negative**. "
    "This app uses an LSTM model trained on the IMDB dataset."
)

model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    review_input = st.text_area(
        "Enter your review here:",
        height=150,
        placeholder="e.g., 'This movie was absolutely fantastic! The acting was superb and the plot was gripping.'"
    )

    if st.button("Analyze Sentiment", type="primary"):
        with st.spinner("Analyzing..."):
            sentiment, score = predict_sentiment(review_input, model, tokenizer)
            score = float(score)
        
        if sentiment:
            st.write("---")
            st.subheader("Analysis Result")
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment} 👍")
                st.progress(score)
                st.metric(label="Confidence Score", value=f"{score:.2%}")
            else:
                st.error(f"Sentiment: {sentiment} 👎")
                st.progress(1 - score) # Show how 'negative' it is
                st.metric(label="Confidence Score", value=f"{1-score:.2%}")
        else:
            st.warning("Please enter a review to analyze.")
else:
    st.error("Model and tokenizer could not be loaded. Please check the file paths and ensure the files are not corrupted.")