import gradio as gr
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained models and vectorizer
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
    "K Nearest Neighbors": joblib.load("k_nearest_neigbours_model.joblib"),
    "Decision Tree": joblib.load("decision_tree_model.joblib"),
    "Random Forest": joblib.load("random_forest_model.joblib"),
    "Support Vector Machine": joblib.load("support_vector_machine_model.joblib"),
    "Multinomial Naive Bayes": joblib.load("multinomial_naive_bayes_model.joblib"),
}
vectorizer = joblib.load("vectorizer.joblib")

# Pre-calculated accuracies (replace with actual accuracy values)
accuracies = {
    "Logistic Regression": "87.37%",
    "K Nearest Neighbors": "62.32%",
    "Decision Tree": "75.39%",
    "Random Forest": "78.43%",
    "Support Vector Machine": "87.91%",
    "Multinomial Naive Bayes": "85.54%"
}

# Prediction function
def test_review(review_text, model_name):
    # Vectorize input review
    review_vectorized = vectorizer.transform([review_text])
    
    # Predict using selected model
    model = models[model_name]
    prediction = model.predict(review_vectorized)[0]
    label = "Original Review" if prediction == 0 else "Computer Generated/Fake Review"
    
    # Fetch accuracy for the selected model
    accuracy = accuracies.get(model_name, "N/A")
    
    return label, f"{model_name} Accuracy: {accuracy}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; color: #4A90E2;'>Fake Review Detection System</h1>")
    
    with gr.Row():
        review_text = gr.Textbox(
            label="Enter Review Text", 
            placeholder="Type a review here...", 
            lines=3
        )
        model_name = gr.Dropdown(
            choices=list(models.keys()), 
            label="Select Model for Prediction", 
            value="Logistic Regression"
        )
        
    with gr.Row():
        predict_btn = gr.Button("Predict Review", variant="primary")
    
    with gr.Row():
        result_text = gr.Textbox(
            label="Prediction Result", 
            placeholder="Prediction will appear here...", 
            interactive=False
        )
        accuracy_text = gr.Textbox(
            label="Model Accuracy", 
            placeholder="Accuracy will appear here...", 
            interactive=False
        )
    
    # Link function to button click
    predict_btn.click(
        fn=test_review, 
        inputs=[review_text, model_name], 
        outputs=[result_text, accuracy_text]
    )

# Launch the Gradio interface
demo.launch()
