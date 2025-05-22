import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Step 1: Read the .txt file and extract image paths and labels
txt_file_path = r"C:\Users\kesha\test venv\one-indexed-files-notrash_test.txt"  # Update this path to your file
img_path = r"C:\Users\kesha\test venv\metal33.jpg"


data = []

# Open and read the .txt file
with open(txt_file_path, 'r') as file:
    lines = file.readlines()


print(f"Number of lines in file: {len(lines)}")
print("First few lines in file:")
print(lines[:431])  

for line in lines:
    # Skip empty lines if there are any
    if line.strip() == "":
        continue
    
    try:
        img_path, label = line.strip().split()  
        data.append((img_path, int(label)))  
    except ValueError:
        print(f"Skipping line due to formatting issue: {line}")

# Step 2: Define the CustomDataset class to load and preprocess images
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return img, label

# Step 3: Define the transformations (resize, convert to tensor, normalize)
transformations = transforms.Compose([
    transforms.Resize((256, 256)),    # Resize images to 256x256
    transforms.ToTensor(),            # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Step 4: Create the dataset and dataloader instances
dataset = CustomDataset(data, transform=transformations)

# Create DataLoader to load data in batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example: Loop through DataLoader to get a batch of images and labels
for imgs, labels in dataloader:
    print(f"Batch of images shape: {imgs.shape}")
    print(f"Batch of labels: {labels}")
    break  # Exit after printing the first batch
