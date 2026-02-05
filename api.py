from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn.functional as F
from src.dataset import get_transform
from src.model import deepfake_model
from PIL import Image
import io

CLASS_NAMES = ["Fake", "Real"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = get_transform(test=True)

model = deepfake_model()
model.to(DEVICE)
model.load_state_dict(torch.load("src/deepfake_model.pth"))
model.eval()



app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "Deepfake Detection API running"}

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return {
        "prediction": CLASS_NAMES[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }