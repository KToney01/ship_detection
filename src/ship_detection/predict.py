import torch

from ship_detection.data import load_image_tensor

CLASS_NAMES = {
    0: "no_ship",
    1: "ship"
}

def predict_image(
        model,
        image_path,
        device
):
    image_tensor = load_image_tensor(image_path)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        return {
            "class_id": predicted_class,
            "label": CLASS_NAMES[predicted_class],
            "confidence": confidence
        }