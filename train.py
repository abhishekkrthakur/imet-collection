from __future__ import print_function
from __future__ import division
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
from utils import CollectionsDataset
from model import train_model
import os
import argparse
from apex import amp
from nnet import model_ft

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=-1)
args = parser.parse_args()

BASE_DIR = "../input/"
FOLD = int(args.fold)
if FOLD == -1:
    FOLD = 0

if FOLD == 0:
    training_folds = [1, 2, 3, 4]
    val_folds = [0]
elif FOLD == 1:
    training_folds = [0, 2, 3, 4]
    val_folds = [1]
elif FOLD == 2:
    training_folds = [0, 1, 3, 4]
    val_folds = [2]
elif FOLD == 3:
    training_folds = [0, 1, 2, 4]
    val_folds = [3]
else:
    training_folds = [0, 1, 2, 3]
    val_folds = [4]

FOLD_NAME = "fold{0}".format(FOLD)

MODEL_NAME = os.environ["MODEL_NAME"]
TRAINING_BATCH_SIZE = int(os.environ["TRAINING_BATCH_SIZE"])
TEST_BATCH_SIZE = int(os.environ["TEST_BATCH_SIZE"])
NUM_CLASSES = int(os.environ["NUM_CLASSES"])
IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])
EPOCHS = int(os.environ["EPOCHS"])

if not os.path.exists(FOLD_NAME):
    os.makedirs(FOLD_NAME)

device = torch.device("cuda:0")
IMG_MEAN = model_ft.mean
IMG_STD = model_ft.std

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

train_dataset = CollectionsDataset(csv_file='../input/folds.csv',
                                   root_dir='../input/train/',
                                   num_classes=NUM_CLASSES,
                                   image_size=IMAGE_SIZE,
                                   folds=training_folds,
                                   transform=train_transform)

valid_dataset = CollectionsDataset(csv_file='../input/folds.csv',
                                   root_dir='../input/train/',
                                   num_classes=NUM_CLASSES,
                                   image_size=IMAGE_SIZE,
                                   folds=val_folds,
                                   transform=val_transform)

train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=TRAINING_BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=4)

valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=4)

model_ft = model_ft.to(device)

lr_min = 1e-4
lr_max = 1e-3

plist = [{'params': model_ft.layer0.parameters()},
         {'params': model_ft.layer1.parameters()},
         {'params': model_ft.layer2.parameters()},
         {'params': model_ft.layer3.parameters(), 'lr': lr_min},
         {'params': model_ft.layer4.parameters(), 'lr': lr_min, 'weight': 0.001},
         {'params': model_ft.last_linear.parameters(), 'lr': lr_max}
         ]

optimizer_ft = optim.Adam(plist, lr=0.001)
model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft, opt_level="O1", verbosity=0)
lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer_ft, verbose=True, factor=0.3, mode="max", patience=1, threshold=0.01)

dataset_sizes = {}
dataset_sizes["train"] = len(train_dataset)
dataset_sizes["val"] = len(valid_dataset)

data_loader = {}
data_loader["train"] = train_dataset_loader
data_loader["val"] = valid_dataset_loader

model_ft = train_model(model_ft,
                       data_loader,
                       dataset_sizes,
                       device,
                       optimizer_ft,
                       lr_sch,
                       num_epochs=EPOCHS,
                       fold_name=FOLD_NAME,
                       use_amp=True)
torch.save(model_ft.state_dict(), os.path.join(FOLD_NAME, "model.bin"))
