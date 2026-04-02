import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

CLASSES = [
    "complex1","complex2","complex3","complex4",
    "blurred","rain","fog","night",
    "occluded","truncated","lp_blurred"
]

# ── Load thresholds ──
with open("outputs/thresholds.json") as f:
    thresholds = json.load(f)

# ── Load model ──
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load("outputs/best_model.pth", map_location="cpu"))
model.eval()

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def predict(img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().numpy()

    result = {}
    for cls, prob in zip(CLASSES, probs):
        t = thresholds.get(cls, 0.5)
        result[cls] = {"prob": round(float(prob), 3), "label": int(prob > t)}

    return result

# ── Test ──
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    out = predict(img_path)
    print(f"\nPredictions for: {img_path}\n")
    for cls, v in out.items():
        bar = "█" * int(v['prob'] * 20)
        tag = "✅" if v['label'] else "  "
        print(f"  {tag} {cls:<20} {v['prob']:.3f}  {bar}")
