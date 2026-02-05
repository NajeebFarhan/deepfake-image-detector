import streamlit as st
import torch
import torch.nn.functional as F
from src.dataset import get_transform
from src.model import deepfake_model
from PIL import Image

CLASS_NAMES = ["Fake", "Real"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = get_transform(test=True)

model = deepfake_model()
model.to(DEVICE)
model.load_state_dict(torch.load("src/deepfake_model.pth"))
model.eval()

st.title("Deepfake & AI-generated Image Detection")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with st.spinner("Loading model...", show_time=True):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        label = CLASS_NAMES[pred.item()]
        confidence = confidence.item() * 100

        st.subheader("Prediction")
        st.write(f"**Result:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")