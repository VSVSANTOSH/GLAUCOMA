import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("glaucoma_model.h5")
    return model

# Load test data 
@st.cache_data
def load_test_data():
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    return X_test, y_test

# Streamlit interface
def main():
    st.title("Glaucoma Detection System")
    
    # Load model and test data
    model = load_model()
    X_test, y_test = load_test_data()

    # Evaluate model 
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Display model performance
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
    with col2:
        st.metric("Test Loss", f"{test_loss:.4f}")

    # File uploader
    st.header("Upload Eye Image for Prediction")
    uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (224, 224))
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display uploaded image
        st.image(img_display, caption="Uploaded Image", width=300)
        
        # Prepare image for prediction
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        is_glaucoma = confidence < 0.5
        
        # Display results
        st.header("Diagnosis Result")
        if is_glaucoma:
            st.error("Diagnosis: Glaucoma")
            st.write(f"Confidence: {(1-confidence)*100:.2f}%")
        else:
            st.success("Diagnosis: Healthy")
            st.write(f"Confidence: {confidence*100:.2f}%")

    # Show some example prediction
    st.header("Example Predictions from Test Set")
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int)
    
    for i in range(min(5, len(X_test))):
        col1, col2 = st.columns([1, 2])
        with col1:
            # Convert normalized float image back to uint8
            img_uint8 = (X_test[i] * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, width=150)
        with col2:
            actual = "Healthy" if y_test[i] == 1 else "Glaucoma"
            predicted = "Healthy" if predicted_classes[i] == 1 else "Glaucoma"
            conf = predictions[i][0] if predicted_classes[i] == 1 else 1-predictions[i][0]
            st.write(f"Actual: {actual}")
            if predicted == "Glaucoma":
                st.error(f"Predicted: {predicted}")
            else:
                st.success(f"Predicted: {predicted}")
            st.write(f"Confidence: {conf*100:.2f}%")
            st.write(f"Correct: {actual == predicted}")
        st.write("---")

if __name__ == "__main__":
    main()