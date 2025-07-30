import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import pathlib
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.optim.lr_scheduler import StepLR
from ptflops import get_model_complexity_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

train_dir = pathlib.Path('D:\LAB_AI\dataset_v1\dataset\\train')
val_dir = pathlib.Path('D:\LAB_AI\dataset_v1\dataset\\test')

train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)                      
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(val_dataset)}")

class InitialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Conv3x1Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Conv1x3Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class RowTemporalWeight(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.params = 0


    def forward(self, x):
        batch, c, h, w = x.shape
        weights = torch.norm(x, dim=1, keepdim=True)
        weights = F.softmax(weights, dim=2)
        weighted_sum = (x * weights).sum(dim=2, keepdim=True)
        weight_sum = weights.sum(dim=2, keepdim=True)
        compressed = weighted_sum / (weight_sum + 1e-8)
        out = compressed.repeat(1, 1, h, 1)
        return out

class ColumnFreqWeight(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.params = 0

    def forward(self, x):
        batch, c, h, w = x.shape
        weights = torch.norm(x, dim=1, keepdim=True)
        weights = F.softmax(weights, dim=3)
        weighted_sum = (x * weights).sum(dim=3, keepdim=True)
        weight_sum = weights.sum(dim=3, keepdim=True)
        compressed = weighted_sum / (weight_sum + 1e-8)
        out = compressed.repeat(1, 1, 1, w)
        return out

class FinalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        adjusted_out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, adjusted_out_channels, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(adjusted_out_channels)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class PatchCorrelationSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.patch_extractor = nn.Unfold(kernel_size=(3, 3), padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if x is None or x.numel() == 0:
            raise ValueError("Đầu vào không hợp lệ")
        batch, c, h, w = x.shape
        patches = self.patch_extractor(x)  # (batch, c*9, h*w)
        patches = patches.view(batch, c, 9, h, w)  # (batch, c, 9, h, w)
        corr = patches.mean(dim=2)  # (batch, c, h, w)
        global_feat = self.global_pool(corr)  # (batch, c, 1, 1)
        attn = self.mlp(global_feat.squeeze(-1).squeeze(-1))  # (batch, c)
        attn = attn.unsqueeze(-1).unsqueeze(-1)  # (batch, c, 1, 1)
        attn = attn.repeat(1, 1, h, w)  
        return x * attn

class TemporalFrequencySelectiveDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(TemporalFrequencySelectiveDownsampling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.temp_gconv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1),
                                    padding=(1, 0), stride=2, bias=False, groups=groups)
   
        self.freq_gconv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3),
                                    padding=(0, 1), stride=2, bias=False, groups=groups)

        self.bn_temp = nn.BatchNorm2d(in_channels)
        self.bn_freq = nn.BatchNorm2d(in_channels)
        self.relation_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        batch, c, h, w = x.shape

        temp_out = self.temp_gconv(x)  # (batch, c, h/2, w)
        temp_out = F.relu(self.bn_temp(temp_out))  
        freq_out = self.freq_gconv(x)  # (batch, c, h, w/2)
        freq_out = F.relu(self.bn_freq(freq_out))  


        h_new = temp_out.shape[2] 
        w_new = freq_out.shape[3]  
        temp_out = temp_out[:, :, :h_new, :w_new]
        freq_out = freq_out[:, :, :h_new, :w_new]  

        spatial_combined = torch.cat([temp_out, freq_out], dim=1) 
        relation_out = self.relation_conv(spatial_combined) 

        combined = torch.cat([spatial_combined, relation_out], dim=1)

        out = self.conv1x1(combined)  

        residual = x[:, :, :h_new, :w_new]  
        residual = self.res_conv(residual) 
        out = out + residual
        out = F.relu(self.bn(out))

        return out

class DTFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_time = Conv3x1Layer(in_channels, out_channels)
        self.conv_freq = Conv1x3Layer(in_channels, out_channels)
        self.rtw = RowTemporalWeight(out_channels)
        self.cfw = ColumnFreqWeight(out_channels)
        self.final_conv = FinalConvLayer(out_channels * 2, out_channels)
        self.pca = PatchCorrelationSpatialAttention(channels=out_channels)

    def forward(self, x):

        x_input = x.requires_grad_(True) 
        time_out = self.conv_time(x)
        time_out = self.rtw(time_out)
        freq_out = self.conv_freq(x)
        freq_out = self.cfw(freq_out)
        out = torch.cat([time_out, freq_out], dim=1)
        out = self.final_conv(out)
        out = out + x_input  
        out = self.pca(out)  
        return out

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TiFiNet(nn.Module):

    def __init__(self, num_classes=12):
        super().__init__()
        self.initial_layer = InitialConvLayer(in_channels=3, out_channels=8)  # (batch, 8, 112, 112)
        self.dtfe1 = DTFEBlock(in_channels=8, out_channels=8)  # (batch, 8, 112, 112)
        self.tfsd1 = TemporalFrequencySelectiveDownsampling(in_channels=8, out_channels=16)  # (batch, 16, 56, 56)
        self.dtfe2 = DTFEBlock(in_channels=16, out_channels=16)  # (batch, 16, 56, 56)
        self.tfsd2 = TemporalFrequencySelectiveDownsampling(in_channels=16, out_channels=24)  # (batch, 24, 28, 28)
        self.dtfe3 = DTFEBlock(in_channels=24, out_channels=24)  # (batch, 24, 28, 28)
        self.tfsd3 = TemporalFrequencySelectiveDownsampling(in_channels=24, out_channels=32)  # (batch, 32, 14, 14)
        self.dtfe4 = DTFEBlock(in_channels=32, out_channels=32)  # (batch, 32, 14, 14)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.initial_layer(x)  # (batch, 8, 112, 112)
        x = self.dtfe1(x)  # (batch, 8, 112, 112)
        x = self.tfsd1(x)  # (batch, 16, 56, 56)
        x = self.dtfe2(x)  # (batch, 16, 56, 56)
        x = self.tfsd2(x)  # (batch, 24, 28, 28)
        x = self.dtfe3(x)  # (batch, 24, 28, 28)
        x = self.tfsd3(x)  # (batch, 32, 14, 14)
        x = self.dtfe4(x)  # (batch, 32, 14, 14)
        x = self.global_pool(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)
        x = self.classifier(x)  # (batch, num_classes)
        return x


num_classes = 12
model = TiFiNet(num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
print(f"Number of trainable parameters: {trainable_params}")

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Train") as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=running_loss/total, accuracy=f"{correct / total:.4f}")
            pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Validate") as pbar:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=running_loss / total, accuracy=f"{correct / total:.4f}")
                pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=1/5)

start_epoch = 0
num_epochs = 60
best_acc = 0.0
best_epoch = start_epoch

resume_train = 'F:\THANHDAT\final_model_epoch_53_acc_0.9020.pt'
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

if os.path.exists(resume_train):
    checkpoint = torch.load(resume_train, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint.get('val_acc', 0.0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    print(f"Resumed training from epoch {start_epoch} with val_acc: {best_acc:.4f}")
else:
    print(f"Checkpoint {resume_train} not found. Starting from scratch.")

# Train and validate
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    current_lr = optimizer.param_groups[0]['lr']

    start_time = time.time()
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    elapsed = time.time() - start_time

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
    print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    print(f"time/epoch: {elapsed:.1f}s, current LR: {current_lr}")

    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_path = f"best_model_epoch_{epoch+1}_acc_{best_acc:.4f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f" Saved best model at epoch {epoch+1} with val_acc: {best_acc:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    final_model_path = f"final_model_epoch_{epoch+1}_acc_{val_acc:.4f}.pt"
    torch.save(checkpoint, final_model_path)
    print(f" Saved final checkpoint at epoch {epoch+1} with val_acc: {val_acc:.4f}")
