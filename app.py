
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.datasets import load_breast_cancer

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

# ---------------------------------
# Load Model, Scaler and Dataset
# ---------------------------------

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "breast_cancer_ann.keras"
    )

    scaler = joblib.load("scaler.pkl")

    dataset = load_breast_cancer()

    return model, scaler, dataset

model, scaler, dataset = load_artifacts()

feature_names = dataset.feature_names
default_values = dataset.data.mean(axis=0)

# ---------------------------------
# Application Title
# ---------------------------------

st.title("🎗️ Breast Cancer Prediction Using ANN")

st.write(
    "An Artificial Neural Network-based application "
    "to predict whether a breast tumor is Benign or Malignant."
)

st.info(
    "Enter the 30 tumor measurement features below "
    "to get a prediction."
)

st.divider()

# ---------------------------------
# User Input
# ---------------------------------

st.subheader("Enter Tumor Characteristics")

user_inputs = {}

with st.form("prediction_form"):

    st.write("Provide the following 30 feature values:")

    columns = st.columns(3)

    for i, feature in enumerate(feature_names):

        with columns[i % 3]:

            user_inputs[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                min_value=0.0,
                value=float(default_values[i]),
                format="%.4f"
            )

    submit = st.form_submit_button(
        "Predict Breast Cancer",
        use_container_width=True
    )

# ---------------------------------
# Prediction
# ---------------------------------

if submit:

    input_data = np.array(
        [list(user_inputs.values())]
    )

    # Standardize input data
    input_scaled = scaler.transform(input_data)

    # ANN Prediction
    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    # Get predicted class
    predicted_label = int(np.argmax(prediction[0]))

    st.divider()

    st.subheader("Prediction Result")

    if predicted_label == 0:

        st.error(
            "⚠️ Malignant Tumor Detected"
        )

        st.write(
            "The ANN model classified the input as "
            "**Malignant (Class 0)**."
        )

    else:

        st.success(
            "✅ Benign Tumor Detected"
        )

        st.write(
            "The ANN model classified the input as "
            "**Benign (Class 1)**."
        )

    # Display model output scores
    st.subheader("Model Output Scores")

    result_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Model Output": prediction[0]
    })

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are ANN output scores, not calibrated "
        "medical probabilities."
    )

    st.warning(
        "This application is for educational purposes only. "
        "It is not a medical diagnosis. Please consult a "
        "qualified healthcare professional for medical evaluation."
    )

# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "Developed using Python, TensorFlow, Keras, "
    "Scikit-learn and Streamlit."
)
