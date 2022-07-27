import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def train_model(train_loader, model, optimizer):
    model.train()
    train_accs = []
    for train_data in train_loader:
        optimizer.zero_grad()
        x, y_true = train_data
        y_pred = model(x)
        train_loss = nn.MSELoss()(y_pred, y_true)
        train_loss.backward()
        optimizer.step()
        train_acc = F.l1_loss(y_pred[:, 0], y_true[:, 0])
        train_accs.append(train_acc)
    return sum(train_accs) / len(train_accs)

def test_model(test_loader, model):
    model.eval()
    test_accs = []
    with torch.no_grad():
        for test_data in test_loader:
            x, y_true = test_data
            y_pred = model(x)
            test_acc = F.l1_loss(y_pred[:, 0], y_true[:, 0])
            test_accs.append(test_acc)
    return sum(test_accs) / len(test_accs)

def predict(loader, model):
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for data in loader:
            x, y_true = data
            y_pred = model(x)
            y_trues.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
    return np.array(y_trues), np.array(y_preds)
