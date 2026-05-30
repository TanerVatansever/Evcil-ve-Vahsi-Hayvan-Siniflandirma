import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split # stratified split için eklendi

from google.colab import drive
drive.mount('/content/drive')

# ---------------------------------------------------------
# 1. KONFİGÜRASYON (CONFIG) - Tüm Parametreler Buradan Yönetilir
# ---------------------------------------------------------
class Config:
    # Veri ve Yollar
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/taner_vatansever/dataset' # Resimlerin bulunduğu ana klasör
    MODEL_SAVE_PATH = 'best_model.pth'

    # Veri Seti Bölme Oranları
    TRAIN_SPLIT = 0.70
    VALID_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Görsel Parametreleri
    IMG_SIZE = 300
    BATCH_SIZE = 32

    # Model Mimarisi Parametreleri
    NUM_CLASSES = 5
    HIDDEN_LAYER_1_NEURONS = 360 # eski değer 512
    HIDDEN_LAYER_2_NEURONS = 180 # eski değer 256
    DROPOUT_RATE = 0.5

    # Eğitim ve Optimizasyon Parametreleri
    EPOCHS = 50
    PATIENCE = 7 # Early Stopping için bekleme süresi
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-5 # L2 Regularization
    OPTIMIZER_TYPE = 'Adam' # 'Adam' veya 'SGD'

    # Donanım
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. VERİ YÜKLEME VE BÖLME
# ---------------------------------------------------------
def get_dataloaders():
    # Görselleri 300x300 boyutuna getirip Tensor'a çevirme ve normalize etme
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=Config.DATA_DIR, transform=transform)
    total_size = len(full_dataset)
    class_names = full_dataset.classes
    print(f"Sınıflar: {class_names} | Toplam Veri: {total_size}")

    # Dinamik ve Stratified Bölme İşlemi
    # İlk olarak eğitim setini ve kalanını ayır
    train_idx, temp_idx, _, temp_labels = train_test_split(
        range(total_size), full_dataset.targets,
        test_size=(Config.VALID_SPLIT + Config.TEST_SPLIT),
        stratify=full_dataset.targets,
        random_state=42
    )

    # Kalanı validasyon ve test setlerine ayır
    valid_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels,
        test_size=(Config.TEST_SPLIT / (Config.VALID_SPLIT + Config.TEST_SPLIT)),
        stratify=temp_labels,
        random_state=42
    )

    # Veri setlerini oluştur
    train_dataset = Subset(full_dataset, train_idx)
    valid_dataset = Subset(full_dataset, valid_idx)
    test_dataset = Subset(full_dataset, test_idx)

    print(f"Train Boyutu: {len(train_dataset)} | Valid Boyutu: {len(valid_dataset)} | Test Boyutu: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader, class_names

# ---------------------------------------------------------
# 3. MODEL TANIMLAMASI (Custom CNN)
# ---------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes, img_size, hidden_layer_1, hidden_layer_2, dropout_rate):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # Output: img_size x img_size x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # Output: img_size/2 x img_size/2 x 32 (e.g., 150x150x32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: img_size/2 x img_size/2 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # Output: img_size/4 x img_size/4 x 64 (e.g., 75x75x64)

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Output: img_size/4 x img_size/4 x 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)     # Output: img_size/8 x img_size/8 x 128 (e.g., 37x37x128 for IMG_SIZE=300)
        )

        # Calculate the size of the flattened layer
        # For IMG_SIZE=300, after 3 MaxPool2d layers (stride 2), image size becomes 300 / (2*2*2) = 300 / 8 = 37.5, which is 37
        # So, 37 * 37 * 128 (last conv output channels)
        self._to_linear = (img_size // 8) * (img_size // 8) * 256 # e.g. (300//8)*(300//8)*128 = 37*37*128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, hidden_layer_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_2, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

def create_model():
    model = CustomCNN(
        num_classes=Config.NUM_CLASSES,
        img_size=Config.IMG_SIZE,
        hidden_layer_1=Config.HIDDEN_LAYER_1_NEURONS,
        hidden_layer_2=Config.HIDDEN_LAYER_2_NEURONS,
        dropout_rate=Config.DROPOUT_RATE
    )
    return model.to(Config.DEVICE)

# ---------------------------------------------------------
# 4. EARLY STOPPING SINIFI
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # Changed from np.Inf to np.inf
        self.path = path
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"   --> Validation loss azaldı ({self.val_loss_min:.6f} --> {val_loss:.6f}). Model '{self.path}' olarak kaydediliyor.")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ---------------------------------------------------------
# 5. EĞİTİM DÖNGÜSÜ
# ---------------------------------------------------------
def train_model(model, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss()

    if Config.OPTIMIZER_TYPE == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9, weight_decay=Config.WEIGHT_DECAY)

    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True, path=Config.MODEL_SAVE_PATH)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, Config.EPOCHS + 1):
        # TRAIN FAZI
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train.double() / total_train

        # VALIDATION FAZI
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val.double() / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_acc'].append(epoch_val_acc.item())

        print(f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        early_stopping(epoch_val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"!!! Early stopping tetiklendi. Eğitim Epoch {epoch}'da durduruluyor. !!!")
            break

    # En iyi modeli yükle
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    return model, history, early_stopping.best_epoch

# ---------------------------------------------------------
# 6. TEST VE DEĞERLENDİRME
# ---------------------------------------------------------
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("\n--- TEST SETİ SONUÇLARI ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("---------------------------\n")

    return all_labels, all_preds, np.array(all_probs)

# ---------------------------------------------------------
# 7. GRAFİK ÇİZİMLERİ
# ---------------------------------------------------------
def plot_metrics(history, best_epoch, y_test, y_pred, y_probs, class_names):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(20, 15))

    # 1. Loss vs Epoch (Early Stop çizgisi ile)
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy vs Epoch
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 3. Learning Curve (Birleştirilmiş Özet)
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.plot(epochs, history['train_acc'], 'b--', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Learning Curve (Loss & Acc)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # 4. Confusion Matrix
    plt.subplot(2, 3, 4)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 5. ROC AUC Curve (Multi-class)
    plt.subplot(2, 3, 5)
    y_test_bin = label_binarize(y_test, classes=range(Config.NUM_CLASSES))
    for i in range(Config.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC AUC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 8. ANA ÇALIŞTIRMA BLOĞU
# ---------------------------------------------------------
if __name__ == '__main__':
    print(f"Kullanılan Donanım: {Config.DEVICE}")

    # Verileri hazırla
    train_loader, valid_loader, test_loader, class_names = get_dataloaders()

    # Modeli oluştur
    model = create_model()

    # Eğitimi başlat
    print("\nEğitim Başlıyor...")
    best_model, history, best_epoch = train_model(model, train_loader, valid_loader)

    # Test setinde değerlendir
    y_test, y_pred, y_probs = evaluate_model(best_model, test_loader, class_names)

    # Grafikleri çiz
    plot_metrics(history, best_epoch, y_test, y_pred, y_probs, class_names)