import joblib
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Load vectorizer and models
vectorizer = joblib.load("vectorizer.joblib")
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
    "K Nearest Neighbors": joblib.load("k_nearest_neighbors_model.joblib"),
    "Decision Tree": joblib.load("decision_tree_model.joblib"),
    "Random Forest": joblib.load("random_forest_model.joblib"),
    "Support Vector Machine": joblib.load("support_vector_machine_model.joblib"),
    "Multinomial Naive Bayes": joblib.load("multinomial_naive_bayes_model.joblib")
}

# Accuracy of models (based on training/test data)
accuracy_scores = {
    "Logistic Regression": "87.37%",
    "K Nearest Neighbors": "62.32%",
    "Decision Tree": "75.39%",
    "Random Forest": "78.43%",
    "Support Vector Machine": "87.91%",
    "Multinomial Naive Bayes": "85.54%"
}

# Visualization and insights
def about_dataset():
    insights = """
    ### Dataset Insights
    - This dataset is used to classify reviews as either "Original" or "Computer Generated."
    - Distribution of classes, average word length, and word frequency are analyzed to better understand the data.
    - Observed Trends:
      - Logistic Regression and SVM achieve the highest prediction accuracy.
      - K Nearest Neighbors has the lowest performance, suggesting a distance-based approach may not capture the review patterns as well.
    """
    return insights

# Function to test a single review with selected model
def test_review(review_text, model_name):
    # Vectorize input review text
    review_vectorized = vectorizer.transform([review_text])
    model = models[model_name]
    
    # Predict with the selected model
    prediction = model.predict(review_vectorized)[0]
    label = "Original Review" if prediction == 0 else "Computer Generated/Fake Review"
    
    # Display prediction and accuracy
    result = {
        "Prediction": label,
        "Model Accuracy": accuracy_scores[model_name]
    }
    return result

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Fake Review Detection System\nAnalyze reviews to detect if they are original or computer-generated.")
    gr.Markdown(about_dataset())

    with gr.Row():
        review_text = gr.Textbox(label="Enter a review to test", placeholder="Type review text here...")
        model_name = gr.Dropdown(choices=list(models.keys()), label="Choose Model")

    predict_btn = gr.Button("Predict")
    output = gr.JSON(label="Prediction Results")

    predict_btn.click(fn=test_review, inputs=[review_text, model_name], outputs=output)

# Run the Gradio app
demo.launch()
