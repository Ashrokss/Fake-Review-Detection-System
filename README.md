# Fake Review Detection System

This is a machine learning-powered web application designed to classify whether a review is *Original* or *Computer Generated*. The app is hosted on Hugging Face Spaces and allows users to input a review, select a classification model, and receive a prediction along with the model's accuracy.

![WhatsApp Image 2024-10-30 at 10 29 22 AM](https://github.com/user-attachments/assets/98bd11a7-a3c9-40e4-8d11-a5c33044ac7b)

---
## Try out here:

click here 👉 [Live link](https://huggingface.co/spaces/Ashish21/Fake-Review-Detection-System)

---

## Features
- **User-Friendly Interface**: A clean, intuitive interface built with Gradio that makes it easy to interact with the model.
- **Multiple Model Choices**: The app offers several machine learning models to classify reviews, including Logistic Regression, Decision Trees, and K-Nearest Neighbors. Users can select a model from a dropdown menu to see how each one performs on their review text.
- **Prediction and Accuracy Display**: After selecting a model and submitting a review, the app will display the prediction result (whether the review is fake or real) and the accuracy score of the selected model.

## Available Models:
- Logistic Regression
- K Nearest Neighbors
- Decision Tree
- Random Forests
- Support Vector Machine
- Multinomial Naive Bayes

## How to Use the App
1. **Enter Review Text**: Type or paste the review you want to classify in the provided text box.
2. **Select Model for Prediction**: Choose a machine learning model from the dropdown menu. Each model may offer a different accuracy level based on the training data.
3. **Predict Review**: Click the **Predict Review** button to generate a classification for the review. The app will display the prediction result along with the selected model's accuracy score.

---

## Tech Stack
- **Gradio**: Used for building the interactive UI.
- **Hugging Face Spaces**: Hosts the application for easy access and public sharing.
- **scikit-learn**: Machine learning library used for model training and evaluation.
- **Joblib**: Employed for loading pre-trained models.

---
 
Explore the app and see how effectively it detects fake reviews!

---
