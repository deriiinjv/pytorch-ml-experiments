import torch
from torchvision import transforms
from PIL import Image
import argparse
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(image_path):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]

    model = get_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return class_names[pred.item()], conf.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    label, confidence = predict(args.image_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
