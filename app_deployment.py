#Pyrhon script for app.py

import gradio as gr
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import torch
import numpy as np

# Load the fine-tuned models
# Sentiment Analysis Model
try:
    sentiment_classifier = pipeline("sentiment-analysis", model="./sentiment_model", tokenizer="./sentiment_model")
except Exception as e:
    print(f"Error loading sentiment model: {e}. Please ensure the 'sentiment_model' folder exists.")
    sentiment_classifier = None

# T5 Summarization Model
try:
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
except Exception as e:
    print(f"Error loading T5 model: {e}. Defaulting to no summarization.")
    t5_model = None

# Sentence-BERT and K-Means for Clustering
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init='auto')
    
    # NOTE: In a real-world scenario, you would save and load the fitted K-Means model.
    # For this demo, we'll quickly fit it to a small sample of the data.
    # You must have 'amazon_reviews_merged.csv' in the same directory.
    df = pd.read_csv('amazon_reviews_merged.csv')
    df = df.rename(columns={'reviews.text': 'text'})
    sample_texts = df['text'].sample(2000, random_state=42).tolist()
    sample_embeddings = embedding_model.encode(sample_texts)
    kmeans.fit(sample_embeddings)
    
except Exception as e:
    print(f"Error setting up clustering: {e}. Skipping clustering functionality.")
    kmeans = None
    embedding_model = None

# --- Define the core functions for the Gradio app ---

def classify_sentiment(text):
    """Classifies the sentiment of a given text."""
    if not sentiment_classifier:
        return "Sentiment model not loaded. Please check your model files."
    result = sentiment_classifier(text)
    return result[0]['label']

def summarize_reviews(reviews_text):
    """Summarizes a block of text containing multiple reviews."""
    if not t5_model:
        return "Summarization model not loaded."
    
    input_text = "summarize: " + reviews_text
    inputs = t5_tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    
    summary_ids = t5_model.generate(
        inputs.input_ids,
        max_length=250,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def predict_cluster(text):
    """Predicts the cluster for a given review text."""
    if not kmeans or not embedding_model:
        return "Clustering model not loaded."
    
    embedding = embedding_model.encode(text)
    cluster_id = kmeans.predict([embedding])[0]
    return f"This review belongs to Cluster {cluster_id}"

# --- Build the Gradio Interface ---

# Interface for Sentiment Analysis
sentiment_interface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter Review Text"),
    outputs="text",
    title="Review Sentiment Classifier",
    description="Classifies a review as positive, negative, or neutral."
)

# Interface for Review Summarization
summarization_interface = gr.Interface(
    fn=summarize_reviews,
    inputs=gr.Textbox(lines=10, label="Enter Reviews to Summarize"),
    outputs="text",
    title="Review Summarizer",
    description="Summarizes multiple reviews into a single coherent text."
)

# Interface for Clustering
clustering_interface = gr.Interface(
    fn=predict_cluster,
    inputs=gr.Textbox(lines=5, label="Enter Review Text to find its category"),
    outputs="text",
    title="Product Category Clustering",
    description="Predicts the product category (cluster) of a new review based on its text."
)

# Combine all interfaces into a single tabbed app
demo = gr.TabbedInterface(
    [sentiment_interface, summarization_interface, clustering_interface],
    ["Sentiment Analysis", "Review Summarization", "Clustering Demo"]
)

# Launch the Gradio app
# The 'share=True' parameter generates a public URL for sharing.
demo.launch(share=True)
