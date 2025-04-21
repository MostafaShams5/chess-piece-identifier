import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from chess_piece_identifier import ChessPiecesIdentifier


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Class names (must match the order used during training)
class_names = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']

@st.cache_resource
def load_model():
    model_path = 'TheTrainedModel.pth'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model first.")
        return None, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessPiecesIdentifier(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Function to make predictions
def predict_chess_piece(image, model, device):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()], probabilities.cpu().numpy()

# Main app
st.title("Chess Piece Identifier")
st.write("Upload an image of a chess piece to identify it")

# Load the model
try:
    model, device = load_model()
    model_loaded = model is not None
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

if model_loaded:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            # Make prediction
            with st.spinner("Identifying chess piece..."):
                prediction, probs = predict_chess_piece(image, model, device)

            st.success(f"Prediction: {prediction}")

            # Show probability distribution
            st.write("Confidence scores:")
            probs_dict = {class_names[i]: float(probs[i]) * 100 for i in range(len(class_names))}
            st.bar_chart(probs_dict)
else:
    st.warning("Please ensure the model is saved at './TheTrainedModel.pth'")

st.markdown("""
### About This App
This app uses a convolutional neural network trained on chess piece images to identify:
- Pawns
- Rooks
- Knights
- Bishops
- Queens
- Kings

For best results, use clear images with good lighting and minimal background clutter.
""")

if __name__ == "__main__":
    
    pass
