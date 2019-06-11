from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import joblib

device = torch.device("cuda:0")
folds = [0, 1, 2, 3, 4]
NUM_CLASSES = 1103

def fbeta_score(y_pred, y_true, thresh, device, beta=2, eps=1e-9, sigmoid=True):
    "Computes the f_beta between `preds` and `targets`"
    y_pred = torch.from_numpy(y_pred).float().to(device)
    y_true = torch.from_numpy(y_true).float().to(device)
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()


def find_best_fixed_threshold(preds, targs, device):
    score = []
    thrs = np.arange(0, 0.5, 0.01)
    for thr in tqdm(thrs):
        score.append(fbeta_score(preds, targs, thresh=thr, device=device))
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print('thr={} F2={}'.format(best_thr, best_score))
    return best_thr, best_score

valid_preds = []
valid_labels = []
test_preds = []
for i in folds:
    if i == 0:
        valid_preds = joblib.load("fold{}/valid_preds.pkl".format(i))
        valid_labels = joblib.load("fold{}/valid_labels.pkl".format(i))
        test_preds = joblib.load("fold{}/test_preds.pkl".format(i))
    else:
        valid_preds = np.vstack((valid_preds, joblib.load("fold{}/valid_preds.pkl".format(i))))
        valid_labels = np.vstack((valid_labels, joblib.load("fold{}/valid_labels.pkl".format(i))))
        test_preds += joblib.load("fold{}/test_preds.pkl".format(i))

test_preds /= len(folds)

best_thr, best_score = find_best_fixed_threshold(valid_preds, valid_labels, device=device)

sample = pd.read_csv("../input/sample_submission.csv")
predicted = []
for i, name in tqdm(enumerate(sample['id'])):
    score_predict = test_preds[i, :].ravel()
    label_predict = np.arange(NUM_CLASSES)[score_predict >= best_thr]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

sample['attribute_ids'] = predicted
sample.to_csv(os.path.join('combined_submission.csv'), index=False)
