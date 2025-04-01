import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentLSTM
from preprocessing import TextPreprocessor
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str):
    """Load and preprocess training data from CSV"""
    df = pd.read_csv(file_path)
    return list(zip(df['text'].tolist(), df['sentiment'].tolist()))

def create_and_train_model():
    # Load training data from CSV
    training_data = load_data('data/train.csv')
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(
        training_data, 
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in training_data]
    )
    
    # Create and fit preprocessor with minimal settings
    all_texts = [text for text, _ in training_data]
    preprocessor = TextPreprocessor(max_vocab_size=1000, max_seq_length=20)
    preprocessor.fit(all_texts)
    
    # Create model
    model = SentimentLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=32,
        hidden_dim=32,
        output_dim=1,
        num_layers=1,
        dropout=0.2,
        bidirectional=True
    )
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to tensors
    X_train = torch.stack([preprocessor.transform(text) for text, _ in train_data])
    y_train = torch.tensor([label for _, label in train_data], dtype=torch.float32)
    
    X_val = torch.stack([preprocessor.transform(text) for text, _ in val_data])
    y_val = torch.tensor([label for _, label in val_data], dtype=torch.float32)
    
    # Training loop
    num_epochs = 100
    batch_size = 16
    best_val_loss = float('inf')
    
    print("Starting training...")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        # Shuffle training data
        indices = torch.randperm(len(train_data))
        
        # Process in batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)
            
            # Calculate validation accuracy
            val_preds = (torch.sigmoid(val_outputs.squeeze()) > 0.5).float()
            val_acc = (val_preds == y_val).float().mean()
        
        model.train()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(indices) / batch_size)
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}\n')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'models/lstm_model.pth')
                preprocessor.save('models/preprocessor.pkl')
    
    # Load best model for testing
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    model.eval()
    
    # Test the model
    print("\nTesting the model:")
    test_texts = [
        "i love you",
        "love",
        "great",
        "i hate you",
        "hate",
        "terrible",
        "this is wonderful",
        "this is horrible",
        "you are amazing",
        "you are awful"
    ]
    
    for text in test_texts:
        with torch.no_grad():
            input_tensor = preprocessor.transform(text).unsqueeze(0)
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            sentiment = "Positive" if probability > 0.5 else "Negative"
            print(f"Text: '{text}' -> {sentiment} (probability: {probability:.4f})")

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    create_and_train_model()
