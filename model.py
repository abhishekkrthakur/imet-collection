import os
import time
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from apex import amp

NUM_CLASSES = int(os.environ["NUM_CLASSES"])


def fbeta_score(y_pred, y_true, thresh, device, beta=2, eps=1e-9, sigmoid=True):
    "Computes the f_beta between `preds` and `targets`"
    y_pred = torch.from_numpy(y_pred).float().to(device)
    y_true = torch.from_numpy(y_true).float().to(device)
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
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


def train_model(model, data_loader, dataset_sizes, device, optimizer, scheduler, num_epochs, fold_name, use_amp=True):
    since = time.time()
    criterion = nn.BCEWithLogitsLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 99999999
    all_scores = []
    best_score = -np.inf
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        valid_preds = np.zeros((dataset_sizes["val"], NUM_CLASSES))
        valid_labels = np.zeros((dataset_sizes["val"], NUM_CLASSES))
        val_bs = data_loader["val"].batch_size
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step(best_score)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            tk0 = tqdm(data_loader[phase], total=int(dataset_sizes[phase] / data_loader[phase].batch_size))
            counter = 0
            for bi, d in enumerate(tk0):
                inputs = d["image"]
                labels = d["labels"]
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if use_amp is True:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                counter += 1
                tk0.set_postfix(loss=(running_loss / (counter * data_loader[phase].batch_size)))

                if phase == "val":
                    valid_labels[bi * val_bs:(bi + 1) * val_bs, :] = labels.detach().cpu().squeeze().numpy()
                    valid_preds[bi * val_bs:(bi + 1) * val_bs, :] = outputs.detach().cpu().squeeze().numpy()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "val":
                best_thr, score = find_best_fixed_threshold(valid_preds, valid_labels, device)
                all_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(fold_name, "model.bin"))

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        if len(all_scores[-5:]) == 5:
            if best_score not in all_scores[-5:]:
                break
            if len(np.unique(all_scores)) == 1:
                break
            if abs(min(all_scores[-5:]) - max(all_scores[-5:])) < 0.001:
                break
        print(all_scores[-5:])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
