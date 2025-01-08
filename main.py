import os
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import *
from utils import *
from models import *


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



DB_BASE_DIR = 'dataset/Train/'
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
NUM_EPOCHS = 50
BATCH_SIZES = [32, 32, 32, 16, 16, 8]
IMAGE_SIZES = [64, 128, 256, 384, 512, 672]
TEST_SIZE = 0.1



df = pd.read_csv('dataset/train.csv')
image_paths, labels = list(df['image']), list(df['label'])
images_train, images_val, labels_train, labels_val = train_test_split(image_paths, labels, test_size=TEST_SIZE, random_state=224)
print('No. Train Images: ', len(images_train), 'No. Val Images: ', len(images_val))



model = MultiHeadAttentionCNN(NUM_CLASSES).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()



# Progressive Upres Training
for IMAGE_SIZE, BATCH_SIZE in zip(IMAGE_SIZES, BATCH_SIZES):
    print(f'Training {model.__class__.__name__}: Image size: {IMAGE_SIZE}, Batch size: {BATCH_SIZE}')
    print('*'*50)
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

    train_dataset = ImageDatasetV2(DB_BASE_DIR, images_train, labels_train, IMAGE_SIZE, train_data_transforms)
    val_dataset = ImageDatasetV2(DB_BASE_DIR, images_val, labels_val, IMAGE_SIZE, test_data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, prefetch_factor=2, num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    save_fname = f'{model.__class__.__name__}_{IMAGE_SIZE}.pt'
    trainer = ModelTrainer(model, train_loader, val_loader, optim, loss_fn, NUM_CLASSES, save_filename=save_fname)

    losses, accs, val_losses, val_accs = trainer.train_model(NUM_EPOCHS, early_stop_patience=15)
    trainer.load_state(f'checkpoints/{model.__class__.__name__}/{save_fname}')

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
    plt.savefig(f'figs/graph_{IMAGE_SIZE}.png')
    plt.close()


print(trainer.validation_epoch())



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
df.to_csv('Submissions/submission_13.csv', index=False)


# !kaggle competitions submit -c cidaut-ai-fake-scene-classification-2024 -f submission_11.csv -m "Submission"


