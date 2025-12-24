import sys
import os
import io
import urllib.request
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
MODEL_URL = os.getenv("MODEL_URL")

os.makedirs(MODEL_DIR, exist_ok=True)

model = None
class_names = None

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        if MODEL_URL is None:
            raise RuntimeError("MODEL_URL environment variable not set")

        print("Downloading model from Hugging Face...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = get_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    yield

app = FastAPI(
    title="Plant Disease Prediction API",
    lifespan=lifespan
)
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return {
        "prediction": class_names[pred.item()],
        "confidence": round(conf.item(), 4)
    }
