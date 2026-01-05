import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torchvision.transforms.functional as TF

# ================= CONFIGURATION =================
# POINT THIS TO YOUR *PERFECT* TRAINING DATA FOLDER
# Example: '/kaggle/input/avenue-dataset/training_videos'
TRAIN_DATA_DIR = '/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/training_videos' 

MODEL_SAVE_PATH = 'rotnet_model.pth'
BATCH_SIZE = 32
EPOCHS = 6      # It learns very fast, 3 is usually enough
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =================================================

class RotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset that creates synthetic training data.
        It takes perfect images and randomly flips them to create 'bad' examples.
        """
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all images in the training folder
        # Adjust extension if your images are .png
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True))
        
        if len(self.image_paths) == 0:
            print(f"WARNING: No images found in {root_dir}. Check your path!")
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # --- SELF-SUPERVISED LOGIC ---
        # 0 = Upright (Original)
        # 1 = Flipped (Rotated 180)
        label = random.choice([0, 1])
        
        if label == 1:
            # Rotate image 180 degrees
            image = image.transpose(Image.FLIP_TOP_BOTTOM)            
        # Apply standard transforms (Resize, Tensorize, Normalize)
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_rotnet():
    print(f"Using Device: {DEVICE}")
    
    # 1. Define Transforms (Standard ResNet sizing)
    # Note: We do NOT use random rotation augmentation here, as that would confuse the labels.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Prepare Data
    full_dataset = RotationDataset(TRAIN_DATA_DIR, transform=transform)
    print(f"Total Frames Found: {len(full_dataset)}")
    
    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Setup Model (ResNet18)
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer for Binary Classification (2 classes: 0 or 1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(DEVICE)
    
    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 20)
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Val Acc:   {val_acc:.2f}%")

    # 6. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print("Step 1 Complete. Ready for cleaning.")

if __name__ == "__main__":
    train_rotnet()