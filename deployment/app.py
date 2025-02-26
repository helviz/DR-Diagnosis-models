from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from timm import create_model

app = Flask(__name__)

# Load the model
model = create_model("deit_base_patch16_224", pretrained=True, num_classes=5)
model.load_state_dict(torch.load("deployment/model.pth"))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
