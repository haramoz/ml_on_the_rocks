import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Setting up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizing the dataset
])

# MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)  # Compressing to 3 dimensions
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate model, loss criterion, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the Autoencoder
def train(model, data_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        for data, _ in data_loader:
            img = data.view(data.size(0), -1).to(device)
            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the Autoencoder
def test(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            img = data.view(data.size(0), -1).to(device)
            output = model(img)
            # You can add more evaluation code here, e.g., comparing original and reconstructed images
            break  # For quick testing, let's look at only one batch
    return img, output

def main():
    train(model, train_loader)
    original, reconstructed = test(model, test_loader)
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axs[0, i].imshow(original[i].view(28, 28).cpu().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed[i].view(28, 28).cpu().numpy(), cmap='gray')
        axs[1, i].axis('off')
    plt.show()

if __name__ == '__main__':
    main()
