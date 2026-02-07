import torch

from src.model import SentimentLSTM

# Load model
model = SentimentLSTM(5000, 50, 128, 1)
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()

# Example text input (simulated as random numbers)
text_tensor = torch.randint(0, 5000, (1, 100))

# Make prediction
with torch.no_grad():
    output = model(text_tensor)
    sentiment = "Positive" if torch.sigmoid(output).item() > 0.5 else "Negative"
    print(f'Sentiment: {sentiment}')
