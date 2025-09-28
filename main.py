from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import gradio as gr

# Load processor & pretrained model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

# Dummy label map (for real accuracy, fine-tune on wet/dry dataset)
label_map = {0: "Dry Waste", 1: "Wet Waste"}

def classify_waste(image):
    # Handle case where no image is uploaded
    if image is None:
        return "⚠️ Please upload an image first!"

    # Convert image to RGB
    img = image.convert("RGB")
    inputs = processor(img, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()

    # Return mapped label (using %2 to force 2 classes for demo)
    return label_map.get(predicted_class % 2, "Unknown")

# Gradio interface
iface = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Wet vs Dry Waste Classifier",
    description="Upload an image of waste and classify it as Wet or Dry."
)

if __name__ == "__main__":
    iface.launch()
