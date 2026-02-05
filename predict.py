import torch
import torch.nn.functional as F
from model import model

def predict_voice(features):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # batch dimension

    with torch.no_grad():
        outputs = model(x)  # shape [1, 2]
        probabilities = F.softmax(outputs, dim=1)  # convert to probabilities

        confidence, prediction = torch.max(probabilities, dim=1)

        confidence = confidence.item()
        prediction = prediction.item()

    if prediction == 1:
        return "AI_GENERATED", round(confidence, 2), "Unnatural pitch consistency detected"
    else:
        return "HUMAN", round(confidence, 2), "Natural human speech patterns"
