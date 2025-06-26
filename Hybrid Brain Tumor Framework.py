
import os
import shutil
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader



original_data_dir = 'C:/Users/Bitto/Tumor/Brain_Tumor_Data_Set'
split_base_dir = 'C:/Users/Bitto/Tumor/Split_Brain'
train_dir = os.path.join(split_base_dir, 'train')
val_dir = os.path.join(split_base_dir, 'val')


def prepare_dataset(original_dir, train_dir, val_dir, split_ratio=0.8):
    if os.path.exists(split_base_dir):
        print("Split dataset already exists.")
        return

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    
    for class_name in ['Healthy', 'Brain_Tumor']:
        os.makedirs(os.path.join(train_dir, class_name))
        os.makedirs(os.path.join(val_dir, class_name))

        images = os.listdir(os.path.join(original_dir, class_name))
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in tqdm(train_images, desc=f"Copying {class_name} train"):
            shutil.copy(os.path.join(original_dir, class_name, img),
                        os.path.join(train_dir, class_name, img))
        
        for img in tqdm(val_images, desc=f"Copying {class_name} val"):
            shutil.copy(os.path.join(original_dir, class_name, img),
                        os.path.join(val_dir, class_name, img))

prepare_dataset(original_data_dir, train_dir, val_dir)


from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as TF

class LaplacianFilter:
    def __call__(self, img):
        
        img_np = np.array(img.convert("L"))
        
        laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(laplacian_rgb)

class GaussianFilter:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        img_np = np.array(img)

        blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(blurred)

class UnsharpMask:
    def __init__(self, kernel_size=5, sigma=1.0, amount=1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount

    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)
        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(img_np, 1 + self.amount, blurred, -self.amount, 0)
        return Image.fromarray(sharpened)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    LaplacianFilter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)


train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)

import numpy as np
from tqdm import tqdm

def extract_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(images.size(0), -1) 
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)


from torchvision import models
import torch.nn.functional as F

mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet_features = mobilenet.features 
mobilenet_features.eval()
mobilenet_features = mobilenet_features.to(device)



import numpy as np
from tqdm import tqdm

def extract_mobilenet_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting MobileNet features"):
            images = images.to(device)
            x = model(images)
            x = F.adaptive_avg_pool2d(x, (1, 1))  
            x = x.view(x.size(0), -1)            
            features.extend(x.cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)



train_features, train_labels = extract_mobilenet_features(train_loader, mobilenet_features)
val_features, val_labels = extract_mobilenet_features(val_loader, mobilenet_features)



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
ab = AdaBoostClassifier()

rf.fit(train_features, train_labels)
knn.fit(train_features, train_labels)
ab.fit(train_features, train_labels)

for name, clf in zip(["Random Forest", "KNN", "AdaBoost"], [rf, knn, ab]):
    preds = clf.predict(val_features)
    acc = accuracy_score(val_labels, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(val_labels, preds))



from torchvision import models


resnet_model = models.resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1]) 
resnet_model.eval()
resnet_model = resnet_model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)


train_features, train_labels = extract_features(train_loader, resnet_model)
val_features, val_labels = extract_features(val_loader, resnet_model)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
ab = AdaBoostClassifier()

rf.fit(train_features, train_labels)
knn.fit(train_features, train_labels)
ab.fit(train_features, train_labels)

for name, clf in zip(["Random Forest", "KNN", "AdaBoost"], [rf, knn, ab]):
    preds = clf.predict(val_features)
    acc = accuracy_score(val_labels, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(val_labels, preds))

import torch.nn.functional as F
import torch.nn as nn

class ScratchedCNN_FeatureExtractor(nn.Module):
    def __init__(self):
        super(ScratchedCNN_FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 56 * 56, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        return x  

model_scratch_feat = ScratchedCNN_FeatureExtractor().to(device)
model_scratch_feat.eval()

def extract_scratch_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting ScratchedCNN features"):
            images = images.to(device)
            outputs = model(images)  
            features.extend(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)

train_features, train_labels = extract_scratch_features(train_loader, model_scratch_feat)
val_features, val_labels = extract_scratch_features(val_loader, model_scratch_feat)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

rf = RandomForestClassifier()
knn = KNeighborsClassifier()
ab = AdaBoostClassifier()

rf.fit(train_features, train_labels)
knn.fit(train_features, train_labels)
ab.fit(train_features, train_labels)

for name, clf in zip(["Random Forest", "KNN", "AdaBoost"], [rf, knn, ab]):
    preds = clf.predict(val_features)
    acc = accuracy_score(val_labels, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(val_labels, preds))

from torchvision import models
import torch.nn.functional as F


vgg19 = models.vgg19(pretrained=True)
vgg19_features = vgg19.features  
vgg19_features.eval()
vgg19_features = vgg19_features.to(device)


def extract_vgg19_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting VGG19 features"):
            images = images.to(device)
            x = model(images)
            x = F.adaptive_avg_pool2d(x, (1, 1))  
            x = x.view(x.size(0), -1)             
            features.extend(x.cpu().numpy())
            labels.extend(targets.numpy())
    return np.array(features), np.array(labels)

train_features_vgg19, train_labels_vgg19 = extract_vgg19_features(train_loader, vgg19_features)
val_features_vgg19, val_labels_vgg19 = extract_vgg19_features(val_loader, vgg19_features)

rf_vgg19 = RandomForestClassifier()
knn_vgg19 = KNeighborsClassifier(n_neighbors=5)
ab_vgg19 = AdaBoostClassifier()

rf_vgg19.fit(train_features_vgg19, train_labels_vgg19)
knn_vgg19.fit(train_features_vgg19, train_labels_vgg19)
ab_vgg19.fit(train_features_vgg19, train_labels_vgg19)


for name, clf in zip(["Random Forest (VGG19)", "KNN (VGG19)", "AdaBoost (VGG19)"], [rf_vgg19, knn_vgg19, ab_vgg19]):
    preds_vgg19 = clf.predict(val_features_vgg19)
    acc_vgg19 = accuracy_score(val_labels_vgg19, preds_vgg19)
    print(f"\n{name} Accuracy: {acc_vgg19:.4f}")
    print(classification_report(val_labels_vgg19, preds_vgg19))
