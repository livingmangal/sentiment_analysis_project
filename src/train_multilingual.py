"""
Training Script for Multilingual Sentiment Analysis
Trains separate models for each language
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.model import SentimentLSTM
from src.preprocessing_multilingual import MultilingualTextPreprocessor


def load_multilingual_data(data_dir: str, language: str) -> List[Tuple[str, int]]:
    """
    Load training data for specific language
    
    Args:
        data_dir: Directory containing language-specific data files
        language: Language code
        
    Returns:
        List of (text, label) tuples
    """
    file_path = os.path.join(data_dir, f'{language}_train.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data not found: {file_path}")

    df = pd.read_csv(file_path)

    # Expect columns: 'text' and 'sentiment' (0 or 1)
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must have 'text' and 'sentiment' columns")

    return list(zip(df['text'].tolist(), df['sentiment'].tolist()))


def train_language_model(
    language: str,
    data_dir: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[float, float]:
    """
    Train model for specific language
    
    Args:
        language: Language code
        data_dir: Directory with training data
        output_dir: Directory to save trained models
        config: Training configuration
    """
    # Default configuration
    if config is None:
        config = {
            'max_vocab_size': 10000,
            'max_seq_length': 30,
            'embedding_dim': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'batch_size': 32,
            'num_epochs': 25,
            'learning_rate': 0.001,
            'weight_decay': 1e-5
        }

    print(f"\n{'='*60}")
    print(f"Training {language.upper()} Sentiment Model")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading {language} training data...")
    training_data = load_multilingual_data(data_dir, language)
    print(f"Loaded {len(training_data)} samples")

    # Split data
    train_data, val_data = train_test_split(
        training_data,
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in training_data]
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create and fit preprocessor
    print(f"\nBuilding {language} vocabulary...")
    preprocessor = MultilingualTextPreprocessor(
        max_vocab_size=config['max_vocab_size'],
        max_seq_length=config['max_seq_length']
    )

    all_texts = [text for text, _ in training_data]
    preprocessor.fit(all_texts, language)

    vocab_size = preprocessor.get_vocab_size(language)
    print(f"Vocabulary size: {vocab_size}")

    # Create model
    print(f"\nInitializing {language} model...")
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=1,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5
    )

    # Prepare data loaders
    print("\nPreparing data loaders...")
    X_train = torch.stack([preprocessor.transform(text, language) for text, _ in train_data])
    y_train = torch.tensor([label for _, label in train_data], dtype=torch.float32)
    X_val = torch.stack([preprocessor.transform(text, language) for text, _ in val_data])
    y_val = torch.tensor([label for _, label in val_data], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.view(-1), batch_y)
                val_loss += loss.item()

                # Calculate accuracy
                predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_dataset)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(output_dir, f'{language}_model.pth')
            model.save(model_path)

            # Save preprocessor
            preprocessor_path = os.path.join(output_dir, f'{language}_preprocessor.pkl')
            preprocessor.save(preprocessor_path)

            # Save metadata
            metadata = {
                "version": "1.0",
                "language": language,
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "val_accuracy": round(val_accuracy, 4),
                "val_loss": round(avg_val_loss, 4),
                "vocab_size": vocab_size,
                "config": config
            }
            metadata_path = os.path.join(output_dir, f'{language}_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

        print()

    print(f"{'='*60}")
    print(f"Training completed for {language.upper()}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")

    return best_val_acc, best_val_loss


def train_all_languages(
    data_dir: str,
    output_dir: str,
    languages: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Train models for all specified languages
    
    Args:
        data_dir: Directory with training data
        output_dir: Directory to save models
        languages: List of language codes (defaults to all supported)
        config: Training configuration
    """
    if languages is None:
        languages = ['en', 'es', 'fr', 'de', 'hi']

    results = {}

    for language in languages:
        try:
            acc, loss = train_language_model(language, data_dir, output_dir, config)
            results[language] = {
                'accuracy': acc,
                'loss': loss,
                'status': 'success'
            }
        except Exception as e:
            print(f"\nError training {language} model: {e}\n")
            results[language] = {
                'status': 'failed',
                'error': str(e)
            }

    # Save training summary
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for lang, result in results.items():
        if result['status'] == 'success':
            print(f"{lang.upper()}: ✓ Accuracy: {result['accuracy']:.4f}")
        else:
            print(f"{lang.upper()}: ✗ Failed - {result.get('error', 'Unknown')}")
    print("="*60)


if __name__ == '__main__':
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_dir = os.path.join(project_root, 'data', 'multilingual')
    output_dir = os.path.join(project_root, 'models', 'multilingual')

    # Custom configuration (optional)
    training_config = {
        'max_vocab_size': 10000,
        'max_seq_length': 30,
        'embedding_dim': 100,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'batch_size': 32,
        'num_epochs': 25,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }

    # Train all languages (or specify subset)
    train_all_languages(
        data_dir=data_dir,
        output_dir=output_dir,
        languages=['en'],  # Start with English, add more as data becomes available
        config=training_config
    )
