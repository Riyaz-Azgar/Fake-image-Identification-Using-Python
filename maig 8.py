import os
import io
from PIL import Image, ImageChops
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def generate_ela(img_path, quality=90):
    original = Image.open(img_path).convert('RGB')
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    compressed = Image.open(buffer)
    ela = ImageChops.difference(original, compressed)
    extrema = ela.getextrema()
    max_diff = max([e[1] for e in extrema])
    if max_diff == 0:
        scale = 1.0
    else:
        scale = 255.0 / max_diff
    ela = ela.point(lambda x: x * scale)
    return ela.resize((128, 128))

class ELAImageDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = generate_ela(self.img_paths[idx])
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train(model, loader, optimizer, criterion, device):
    model.train()
    total, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += labels.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
    acc = correct / total
    return loss.item(), acc

def predict(model, img_path, device):
    model.eval()
    ela = generate_ela(img_path)
    tensor = transforms.ToTensor()(ela).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        label = logits.argmax(1).item()
    return "FAKE" if label == 1 else "REAL"

if __name__ == "__main__":
    train_dataset = ELAImageDataset(img_paths, labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        loss, acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, train_acc={acc:.3f}")

    print(predict(model, "test_image.jpg", device))
