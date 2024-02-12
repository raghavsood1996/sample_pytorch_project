# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Example parameters - adjust based on your dataset
input_size = 11  # Number of input features
hidden_size = 50  # Number of hidden units in each layer
num_classes = 2 # Number of output classes
num_epochs = 500
batch_size = 5
learning_rate = 0.001

# Example synthetic data
# Load the dataset
df = pd.read_csv('../data/loan_data.csv')
df.dropna(inplace=True)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, '' : 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0, '' : 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0, '' : 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0, '' : 0})
df['Dependents'] = df['Dependents'].str.rstrip('+')


# Preprocess the data
# This includes dropping unnecessary columns, handling missing values, etc.
# For example, if 'feature1' and 'feature2' are your feature columns and 'label' is your target
X = df[['Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Property_Area']].values


y = df['Loan_Status'].values

# Optionally encode categorical variables if any
# For binary classification, ensure your labels are encoded as 0 and 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.int64)))
test_dataset = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.int64)))


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = SimpleNN(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        # print(inputs)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Finished Training")
