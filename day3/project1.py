import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Features and labels
X = torch.rand(100, 5)  # 100 samples, 5 features each
y = torch.randint(0, 2, (100,))  # Binary target

# Creating a dataset and dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Input layer
        self.relu = nn.ReLU()        # Activation function
        self.fc2 = nn.Linear(10, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)  # Output activation

model = BinaryClassifier()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test with a new input
test_input = torch.rand(1, 5)
predicted = model(test_input).item() > 0.5
print(f"Predicted class for the input is: {'Class 1' if predicted else 'Class 0'}")
