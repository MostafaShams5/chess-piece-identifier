import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import matplotlib.pyplot as plt

class ChessPiecesIdentifier(nn.Module):
    def __init__(self, NumberPieces):
        super().__init__()
        self.model = nn.Sequential(
    # First convolutional block with batch normalization
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Second convolutional block with increased filters and batch normalization
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # Third convolutional block to extract more features
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),

    # Reduced size of fully connected layer to prevent overfitting
    nn.Linear(128 * 28 * 28, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4), # Add dropout to prevent overfitting

    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, NumberPieces)
)
    def forward(self, x):
        return self.model(x)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Path for saving model
MODEL_SAVE_PATH = 'TheTrainedModel.pth'

def train_model(dataset_path):
    print("Training new model...")

  
    trainData = ImageFolder(root=dataset_path, transform=train_transform)

    
    class_names = trainData.classes
    print(f"Classes: {class_names}")

  
    train_size = int(0.8 * len(trainData))
    test_size = len(trainData) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(trainData, [train_size, test_size])

    trainLoader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessPiecesIdentifier(len(class_names))
    model = model.to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00, weight_decay=1e-5)
    num_epochs = 50

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        if epoch == 45:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

        model.train()
        running_loss = 0.0
        for images, labels in trainLoader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainLoader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return model, device, class_names

def load_or_train_model(dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get class names from dataset
    trainData = ImageFolder(root=dataset_path, transform=train_transform)
    class_names = trainData.classes

    # Check if model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        model = ChessPiecesIdentifier(len(class_names))
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device, class_names
    else:
        return train_model(dataset_path)

def test_with_image(image_path, model, device, class_names):
    # Load image
    img = Image.open(image_path)
    
    # Predict
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    # Print results
    print(f"Predicted: {class_names[predicted.item()]}")
    
    return class_names[predicted.item()], probabilities.cpu().numpy()
dataset_path = '/home/shams/Documents/Chess/Chessman-image-dataset/Chess'
load_or_train_model(dataset_path=dataset_path)
