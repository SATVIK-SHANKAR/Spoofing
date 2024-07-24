import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the custom edge detection transform
class EdgeDetectionTransform:
    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 3:  # Convert color image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 100, 200)
        # Convert single channel to three channels by duplicating the grayscale image
        edges_rgb = np.stack([edges]*3, axis=-1)
        return Image.fromarray(edges_rgb)

# Data augmentation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    EdgeDetectionTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Specify the path to your dataset here
dataset_path = 'path to folder'  # Replace this with the actual path

# Create the dataset and dataloader
train_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load the pretrained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Modify the classifier to fit our binary classification task
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 2)

# Initialize the model, loss function, and optimizer
model = vgg16
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Define the path where you want to save the model
save_dir = 'path to folder'  # Replace with your desired directory
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
model_path = os.path.join(save_dir, 'vgg16_model.pth')

# Save the trained model
torch.save(model.state_dict(), model_path)

# Define the data transformations for prediction
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    EdgeDetectionTransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_image(image_path, model, device):
    model.eval()  # Set the model to evaluation mode
    image = Image.open(image_path).convert('RGB')  # Open the image and convert it to RGB
    image_transformed = data_transforms(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    
    with torch.no_grad():  # Disable gradient computation
        outputs = model(image_transformed)  # Get model outputs
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
    
    return 'real' if predicted.item() == 0 else 'spoof'  # Map prediction to class name

def display_image(image_path, prediction):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

# Load the model from the saved file
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Predict and display images in a folder
input_folder = 'path to folder'  # Replace with the path to your folder of images

for img_name in os.listdir(input_folder):
    if img_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, img_name)
        prediction = predict_image(img_path, model, device)
        display_image(img_path, prediction)

print("Predictions and display completed.")
