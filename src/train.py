import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

def initialize_weights(m):
    """
    Initializes the weights of the PyTorch model.
    Applies Xavier uniform initialization to linear layers and sets biases to zero.
    This helps in keeping the signal from becoming too small or too large during training.
    
    Args:
    m (torch.nn.Module): PyTorch model or layer.
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def load_and_preprocess_data(filepath):
    """
    Loads the dataset from a CSV file, preprocesses it by encoding categorical variables,
    handling missing values, and performing feature scaling.
    
    Args:
    filepath (str): Path to the CSV file containing the dataset.
    
    Returns:
    Tuple of numpy arrays: (X_train, X_test, y_train, y_test)
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    # Encode categorical features
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    }
    df.replace(mappings, inplace=True)
    df['Dependents'] = df['Dependents'].str.rstrip('+').astype(float)

    # Extract features and target variable
    features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    X = df[features].values
    y = LabelEncoder().fit_transform(df['Loan_Status'].values)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    """
    Creates DataLoader objects for the training and test sets.
    
    Args:
    X_train, X_test, y_train, y_test: Numpy arrays containing the split dataset.
    batch_size (int): Batch size for the DataLoader.
    
    Returns:
    Tuple of DataLoader: (train_loader, test_loader)
    """
    # Convert to PyTorch tensors and create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Trains the PyTorch model using the given training DataLoader.
    
    Args:
    model (torch.nn.Module): The PyTorch model to train.
    train_loader (DataLoader): DataLoader for the training data.
    criterion (torch.nn.modules.loss): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer.
    num_epochs (int): Number of epochs to train for.
    """
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader, num_classes):
    """
    Evaluates the trained PyTorch model on the test set.
    
    Args:
    model (torch.nn.Module): The trained PyTorch model.
    test_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes in the dataset.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(num_classes)]
        n_class_samples = [0 for i in range(num_classes)]
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        for i in range(num_classes):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of class {i}: {acc} %')

def main():
    """
    Main function to run the training and evaluation of the model.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('../data/loan_data.csv')
    
    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=4)
    
    # Initialize model, loss, and optimizer
    model = SimpleNN(input_size=11, hidden_size=15, num_classes=2)
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    # Evaluate the model
    evaluate_model(model, test_loader, num_classes=2)

if __name__ == '__main__':
    main()
