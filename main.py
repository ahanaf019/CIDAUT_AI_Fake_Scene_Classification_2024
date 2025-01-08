import os
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from dataset import *
from utils import *
from models import *


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_CLASSES = 2
NUM_EPOCHS = 50





df = pd.read_csv('dataset/train.csv')
image_paths = list(df['image'])
labels = list(df['label'])
base_dir = 'dataset/Train/'

images_train, images_val, labels_train, labels_val = train_test_split(image_paths, labels, test_size=0.15, random_state=224)
len(images_train), len(images_val)



model = MultiHeadAttentionCNN(NUM_CLASSES).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
# scaler = GradScaler()


from torchinfo import summary

# summary(model, input_size=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))


for IMAGE_SIZE in [64, 128, 256, 384, 512]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.02)),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomPerspective(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(),
    ])

    test_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageDatasetV2(base_dir, images_train, labels_train, IMAGE_SIZE, train_data_transforms)
    val_dataset = ImageDatasetV2(base_dir, images_val, labels_val, IMAGE_SIZE, test_data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, prefetch_factor=2, num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses, accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optim, loss_fn, NUM_CLASSES, NUM_EPOCHS, IMAGE_SIZE, device
    )    

    plt.figure(figsize=(12, 6))

    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])

    plt.subplot(1,2,2)
    plt.plot(accs)
    plt.plot(val_accs)
    plt.legend(['acc', 'val_acc'])
    plt.suptitle(f'Image Size: {IMAGE_SIZE}')
    plt.show()


load_state(f'checkpoints/{model.__class__.__name__}/{model.__class__.__name__}_frozen=-5_epoch-31_val-loss-0.0546.pt', model, optim)


validation_epoch(model, val_loader, loss_fn, optim, device)



test_images_paths = sorted(glob('dataset/Test/*.jpg'))
preds = []
fids = []
for path in test_images_paths:
    image = read_image(path, size=IMAGE_SIZE)
    image = test_data_transforms(image).unsqueeze(0).to(device)
    pred = torch.softmax(model(image), dim=1)
    index = torch.argmax(model(image)).item()
    pred = pred[0][1].item()
    fid = Path(path).name
    preds.append(pred)
    fids.append(fid)

df = pd.DataFrame({
    'image': fids,
    'label': preds
})
df.to_csv('submission_11.csv', index=False)
df


# !kaggle competitions submit -c cidaut-ai-fake-scene-classification-2024 -f submission_11.csv -m "Submission"


