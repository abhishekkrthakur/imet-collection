from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import os
from tqdm import tqdm
import joblib
from utils import CollectionsDataset, CollectionsDatasetTest
from model import find_best_fixed_threshold
import argparse
from nnet import model_ft

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=-1)
args = parser.parse_args()

BASE_DIR = "../input/"
FOLD = int(args.fold)
if FOLD == -1:
    FOLD = 0

MODEL_NAME = os.environ["MODEL_NAME"]
TRAINING_BATCH_SIZE = int(os.environ["TRAINING_BATCH_SIZE"])
TEST_BATCH_SIZE = int(os.environ["TEST_BATCH_SIZE"])
NUM_CLASSES = int(os.environ["NUM_CLASSES"])
IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])
FOLD_NAME = "fold{0}".format(FOLD)

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

device = torch.device("cuda:0")
IMG_MEAN = model_ft.mean
IMG_STD = model_ft.std


test_transform=transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN,IMG_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

valid_dataset = CollectionsDataset(csv_file='../input/folds.csv',
                                   root_dir='../input/train/',
                                   num_classes=NUM_CLASSES,
                                   image_size=IMAGE_SIZE,
                                   folds=val_folds,
                                   transform=val_transform)


valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=4)

test_dataset = CollectionsDatasetTest(csv_file='../input/sample_submission.csv',
                                  root_dir='../input/test/',
                                  image_size=IMAGE_SIZE,
                                  transform=test_transform)

test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=TEST_BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=4)


model_ft.load_state_dict(torch.load(os.path.join(FOLD_NAME, "model.bin")))
model_ft = model_ft.to(device)

for param in model_ft.parameters():
    param.requires_grad = False


model_ft.eval()
valid_preds = np.zeros((len(valid_dataset), NUM_CLASSES))
valid_labels = np.zeros((len(valid_dataset), NUM_CLASSES))
tk0 = tqdm(valid_dataset_loader)
for i, _batch in enumerate(tk0):
    x_batch = _batch["image"]
    y_batch = _batch["labels"]
    pred = model_ft(x_batch.to(device))
    valid_labels[i * TEST_BATCH_SIZE:(i + 1) * TEST_BATCH_SIZE, :] = y_batch.detach().cpu().squeeze().numpy()
    valid_preds[i * TEST_BATCH_SIZE:(i + 1) * TEST_BATCH_SIZE, :] = pred.detach().cpu().squeeze().numpy()

best_thr, best_score = find_best_fixed_threshold(valid_preds, valid_labels, device=device)

test_preds = np.zeros((len(test_dataset), NUM_CLASSES))
tk0 = tqdm(test_dataset_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model_ft(x_batch.to(device))
    test_preds[i * TEST_BATCH_SIZE:(i + 1) * TEST_BATCH_SIZE, :] = pred.detach().cpu().squeeze().numpy()


test_preds = torch.from_numpy(test_preds).float().to(device).sigmoid()
test_preds = test_preds.detach().cpu().squeeze().numpy()

sample = pd.read_csv("../input/sample_submission.csv")
predicted = []
for i, name in tqdm(enumerate(sample['id'])):
    score_predict = test_preds[i, :].ravel()
    label_predict = np.arange(NUM_CLASSES)[score_predict >= best_thr]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

sample['attribute_ids'] = predicted

# save all the stuff
sample.to_csv(os.path.join(FOLD_NAME, 'submission.csv'), index=False)
joblib.dump(valid_preds, os.path.join(FOLD_NAME, "valid_preds.pkl"))
joblib.dump(test_preds, os.path.join(FOLD_NAME, "test_preds.pkl"))
joblib.dump(valid_labels, os.path.join(FOLD_NAME, "valid_labels.pkl"))
