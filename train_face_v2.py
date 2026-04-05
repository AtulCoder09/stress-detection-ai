"""
ADVANCED FACE EMOTION TRAINING SCRIPT (V2)
------------------------------------------
This script uses advanced data augmentation, Label Smoothing, and Mixed Precision
to push the EfficientNet-B2 accuracy as close to 99% as mathematically possible
without overfitting.

Architecture: EfficientNet-B2 + CBAM Attention
Dataset: 7 Emotions (angry, disgust, fear, happy, neutral, sad, surprise)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import copy

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION & HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════
DATA_DIR = r"dataset_face\dataset_face"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODELS_DIR, "face_model_v2.pth")

# Hyperparameters built for high accuracy
BATCH_SIZE = 32
NUM_EPOCHS = 40          # High epochs to utilize the Cosine Annealing scheduler
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4      # L2 Regularization to stop overfitting
NUM_WORKERS = 4 if os.name != 'nt' else 0  # Safe default for Windows

# Automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Training on Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════
#  HEAVY DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════
# By adding random horizontal flips, mild rotations, and color jitter,
# the model is forced to recognize *emotions* underneath lighting/angle changes.

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),           # Flip 50% of images
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ═══════════════════════════════════════════════════════════
#  DATALOADERS
# ═══════════════════════════════════════════════════════════
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val', 'test']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=NUM_WORKERS)
    for x in ['train', 'val', 'test']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
NUM_CLASSES = len(class_names)

print(f"📁 Classes found ({NUM_CLASSES}): {class_names}")
print(f"📊 Dataset Sizes: {dataset_sizes}")

# ═══════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE (EfficientNet-B2 + CBAM)
# ═══════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(),
            nn.Linear(ch // r, ch, bias=False)
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        return torch.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class EmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.backbone = backbone.features
        self.cbam = CBAM(1408)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1408 * 2),
            nn.Linear(1408 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),     # 50% dropout prevents memorizing the training set
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)
        x = torch.cat([avg, mx], dim=1)
        return self.classifier(x)

# ═══════════════════════════════════════════════════════════
#  TRAINING PIPELINE (State-of-the-Art implementation)
# ═══════════════════════════════════════════════════════════
model = EmotionModel(NUM_CLASSES).to(DEVICE)

# 1. Label Smoothing: Prevents the model from being "too confident" (e.g., 99.9% angry), 
#    which forces it to keep learning subtle features for a higher final accuracy.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 2. AdamW Optimizer: Standard Adam, but with mathematically correct Weight Decay.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 3. ReduceLROnPlateau: If the accuracy stops improving for 3 epochs, cut the learning 
#    rate in half to make finer adjustments to the weights.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# 4. Mixed Precision Scaler: Speeds up training on RTX GPUs and saves memory.
scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

def train_model():
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n🚀 Beginning Deep Learning Training...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n[{epoch+1}/{NUM_EPOCHS}] {'-' * 20}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Mixed Precision context
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.upper():<7} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")

            # Update best model
            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save current best to disk
                    torch.save({'model_state_dict': best_model_wts}, SAVE_PATH)
                    print(f"   ⭐ New best validation accuracy! Checkpoint saved to {SAVE_PATH}")

    time_elapsed = time.time() - since
    print(f"\n🎉 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc*100:.2f}%")

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    train_model()
