"""
ADVANCED VOICE EMOTION TRAINING SCRIPT (V2)
-------------------------------------------
This script trains a Wav2Vec2 backbone on your specific 5-class dataset 
(calm, low, moderate, high, extreme). Unlike V1 where the backbone was entirely frozen, 
V2 unfreezes the top 2 transformer layers of Wav2Vec2 so it learns acoustic features 
specific to varying levels of human "stress" rather than generic English speech.

Architecture: Wav2Vec2 (Top 2 layers unfrozen) -> 5-Node MLP Head
Dataset: 5 Stress Intensities (calm, low, moderate, high, extreme)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import time
import copy
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION & HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════
DATA_DIR = r"dataset_voice\split_combined"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODELS_DIR, "voice_model_v2.pth")

SAMPLE_RATE = 16_000
MAX_DURATION = 3.0           # 3-second audio chunks (padded/truncated)
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)

BATCH_SIZE = 16 if torch.cuda.is_available() else 4  # Wav2Vec2 is very RAM heavy
NUM_EPOCHS = 25
LEARNING_RATE = 2e-5      # VERY small LR to safely adjust pretrained Wav2Vec2 weights
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Voice Training on Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════
#  CUSTOM AUDIO DATASET (Added Noise Augmentation)
# ═══════════════════════════════════════════════════════════
# During training, we randomly add white noise. This ensures the 
# high accuracy carries over to real-world deployment on cheap webcams/mics.

class AudioEmotionDataset(Dataset):
    def __init__(self, data_dir, split, processor, augment=False):
        self.split_dir = os.path.join(data_dir, split)
        self.processor = processor
        self.augment = augment
        
        self.filepaths = []
        self.labels = []
        
        # Collect all files
        valid_exts = {".wav", ".mp3", ".m4a"}
        if os.path.exists(self.split_dir):
            for class_name in os.listdir(self.split_dir):
                class_dir = os.path.join(self.split_dir, class_name)
                if not os.path.isdir(class_dir): continue
                for f in os.listdir(class_dir):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        self.filepaths.append(os.path.join(class_dir, f))
                        self.labels.append(class_name)

        # Pre-encode labels
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.classes = list(self.label_encoder.classes_)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels_encoded[idx]
        
        # Load audio (downsample to 16kHz for Wav2Vec2)
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        
        # Augmentation (Training ONLY)
        if self.augment and np.random.rand() < 0.3:
            # Inject slight Gaussian noise (30% chance)
            noise = np.random.randn(len(audio))
            audio = audio + 0.005 * noise
        
        # Padding/Truncating to exactly 3 seconds
        if len(audio) < MAX_LEN:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))
        else:
            audio = audio[:MAX_LEN]
            
        return audio, label

def collate_fn(batch, processor):
    audios = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    
    # Processor handles mean/var normalization automatically
    inputs = processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    return inputs.input_values, torch.tensor(labels, dtype=torch.long)

# ═══════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════
class AdvancedWav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # UNFREEZE THE TOP 2 TRANSFORMER LAYERS to learn emotion-specific features
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        for layer in self.wav2vec2.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)

# ═══════════════════════════════════════════════════════════
#  TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════
def train_model():
    print("📦 Loading Wav2Vec2 Processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    print("\n📂 Scanning Datasets...")
    datasets = {
        'train': AudioEmotionDataset(DATA_DIR, 'train', processor, augment=True),
        'val': AudioEmotionDataset(DATA_DIR, 'val', processor, augment=False)
    }
    
    # We must save the label encoder to disk so app.py knows the 5 classes!
    label_encoder = datasets['train'].label_encoder
    NUM_CLASSES = len(datasets['train'].get_classes())
    print(f"   🎙️ Found Classes ({NUM_CLASSES}): {datasets['train'].get_classes()}")

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), 
                      collate_fn=lambda b: collate_fn(b, processor),
                      num_workers=4 if os.name != 'nt' else 0)
        for x in ['train', 'val']
    }

    model = AdvancedWav2Vec2Classifier(NUM_CLASSES).to(DEVICE)
    
    # Focal Loss prevents the model from being biased toward the dominant class.
    # We stick to standard LabelSmoothing here because of the small LR and unfreezing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Different Learning Rates: High LR for classifier, tiny LR for Wav2Vec2 layers
    classifier_params = list(model.classifier.parameters())
    wav_params = []
    for layer in model.wav2vec2.encoder.layers[-2:]:
        wav_params.extend(list(layer.parameters()))
        
    optimizer = torch.optim.AdamW([
        {'params': classifier_params, 'lr': LEARNING_RATE * 10},
        {'params': wav_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    print("\n🚀 Beginning Voice Model Fine-Tuning...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n[{epoch+1}/{NUM_EPOCHS}] {'-' * 20}")
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            for input_values, labels in dataloaders[phase]:
                input_values = input_values.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                        outputs = model(input_values)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * input_values.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])
            print(f"{phase.upper():<7} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")

            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    
                    # We MUST save the LabelEncoder with the weights so app.py can map output nodes to strings
                    torch.save({
                        'model_state_dict': best_wts,
                        'label_encoder': label_encoder
                    }, SAVE_PATH)
                    print(f"   ⭐ New best voice accuracy! Saved to {SAVE_PATH}")

    time_elapsed = time.time() - since
    print(f"\n🎉 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == '__main__':
    train_model()
