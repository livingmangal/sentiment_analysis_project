import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SentimentLSTM
from preprocessing import TextPreprocessor
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def load_data(file_path: str):
    """Load and preprocess training data from CSV"""
    df = pd.read_csv(file_path)
    # Ensure sentiment column handles new class "2" automatically
    return list(zip(df['text'].tolist(), df['sentiment'].tolist()))

def create_and_train_model():
    # Load training data from CSV
    training_data = load_data(os.path.join(project_root, 'data', 'train.csv'))
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(
        training_data, 
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in training_data]
    )
    
    # Create and fit preprocessor with minimal settings
    all_texts = [text for text, _ in training_data]
    preprocessor = TextPreprocessor(max_vocab_size=5000, max_seq_length=20)
    preprocessor.fit(all_texts)
    
    # Create model
    model = SentimentLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        output_dim=3,  # Changed to 3 for Negative, Positive, Neutral
        num_layers=1,
        dropout=0.2,
        bidirectional=True
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for multi-class
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to tensors and create DataLoaders
    X_train = torch.stack([preprocessor.transform(text) for text, _ in train_data])
    y_train = torch.tensor([label for _, label in train_data], dtype=torch.long) # Changed to long
    X_val = torch.stack([preprocessor.transform(text) for text, _ in val_data])
    y_val = torch.tensor([label for _, label in val_data], dtype=torch.long) # Changed to long
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Training loop
    num_epochs = 15 # Increased epochs slightly for multi-class
    best_val_loss = float('inf')
    
    print("Starting training...")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Process in batches using DataLoader
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) # Removed squeeze
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_outputs = model(batch_X)
                val_loss += criterion(val_outputs, batch_y).item()
                
                # Calculate validation accuracy
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc += (val_preds == batch_y).float().sum().item()
        
        # Average validation metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)
        
        model.train()
        
        # Print progress
        if (epoch + 1) % 1 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}\n')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(project_root, 'models', 'lstm_model.pth'))
                preprocessor.save(os.path.join(project_root, 'models', 'preprocessor.pkl'))
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(project_root, 'models', 'lstm_model.pth')))
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
        "you are awful",
        "it was okay",
        "average experience",
        "nothing special",
        "i don't care"
    ]
    
    sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    
    for text in test_texts:
        with torch.no_grad():
            input_tensor = preprocessor.transform(text).unsqueeze(0)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
            sentiment = sentiment_map.get(pred_idx, "Unknown")
            
            print(f"Text: '{text}' -> {sentiment} (confidence: {confidence:.4f})")

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    create_and_train_model()
