import os
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom dataset to include LBP feature extraction
class FaceSpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.data = []
        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('jpg', 'jpeg', 'png')):
                    self.data.append((os.path.join(class_dir, img_name), class_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        lbp_image = self.extract_lbp(cv2.imread(img_path, 0))
        return (image, lbp_image), label

    @staticmethod
    def extract_lbp(image):
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        lbp = lbp.astype(np.float32) / 255.0  # Normalize to [0, 1]
        lbp = cv2.resize(lbp, (224, 224))  # Resize to match input size of CNN
        lbp = np.expand_dims(lbp, axis=0)  # Add channel dimension
        return torch.from_numpy(lbp).float()

# Data augmentation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Specify the path to your dataset
dataset_path = 'path to folder'

# Create the dataset and dataloader
train_dataset = FaceSpoofDataset(root_dir=dataset_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple CNN model for LBP features
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 112 * 112, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 112 * 112)
        x = F.relu(self.fc1(x))
        return x

# Dual-channel network combining deep features and LBP features
class DualChannelNet(nn.Module):
    def __init__(self):
        super(DualChannelNet, self).__init__()
        self.shallow_cnn = ShallowCNN()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()  # Remove the last fully connected layer
        self.fc = nn.Linear(2048 + 256, 2)  # Combine deep and shallow features

    def forward(self, x):
        image, lbp_image = x
        deep_features = self.resnet50(image)
        shallow_features = self.shallow_cnn(lbp_image)
        combined_features = torch.cat((deep_features, shallow_features), dim=1)
        output = self.fc(combined_features)
        return output

# Training loop with performance metrics
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        
        for (inputs, lbp_images), labels in train_loader:
            inputs, lbp_images, labels = inputs.to(device), lbp_images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model((inputs, lbp_images))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        epoch_precision = precision_score(all_labels, all_predictions, average='weighted')
        epoch_recall = recall_score(all_labels, all_predictions, average='weighted')
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1 Score: {epoch_f1:.4f}")

# Initialize the model, loss function, and optimizer
model = DualChannelNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# Save the trained model
save_dir = 'path to folder'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'dual_channel_model.pth')
torch.save(model.state_dict(), model_path)

# Define the data transformations for prediction
data_transforms_predict = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction and display functions
def predict_image(image_path, model, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_transformed = data_transforms_predict(image).unsqueeze(0).to(device)
    lbp_image = FaceSpoofDataset.extract_lbp(cv2.imread(image_path, 0)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model((image_transformed, lbp_image))
        _, predicted = torch.max(outputs, 1)
    
    return 'real' if predicted.item() == 0 else 'spoof'

def display_image(image_path, prediction):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

# Load the model from the saved file
model = DualChannelNet()
model.load_state_dict(torch.load(model_path))
model.to(device)

# Predict and display images in a folder
input_folder = 'path to folder'

for img_name in os.listdir(input_folder):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, img_name)
        prediction = predict_image(img_path, model, device)
        display_image(img_path, prediction)

print("Predictions and display completed.")
