import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchvision import datasets
import numpy as np

torch.__version__


def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch, (images, labels) in enumerate(data_loader):

        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # print(batch)
        if batch % 15 == 0:
            print(f'Train Batch: {batch}')

    return total_loss / len(data_loader)


def validate_epoch(model, data_loader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for batch, (images, labels) in enumerate(data_loader):

            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            test_pred_labels = outputs.argmax(dim=1)
            test_acc += ((test_pred_labels == labels).sum().item() / len(test_pred_labels))

            if batch % 3 == 0:
                print(f'Test Batch: {batch}')

    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader)
    return test_loss, test_acc


from tqdm.auto import tqdm


def early_stopping_trigger(val_loss, min_loss, patience, counter):
    # If validation loss increases for several epochs, trigger early stopping
    if val_loss < min_loss:
        min_loss = val_loss
        counter = 0
    else:
        counter += 1
    return min_loss, counter, counter >= patience

min_loss = float('inf')
patience = 10
counter = 0

for epoch in tqdm(range(10), desc='Training Model'):
    train_loss = train_epoch('model', 'train_subset_loader', 'criterion', 'optimizer')
    val_loss, val_acc = validate_epoch('model', 'val_subset_loader', 'criterion')
    print(f'Epoch {epoch+1}: Training Loss: {train_loss:.4f} |  Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}')

    # Check for early stopping if validation loss consistently increases
    min_loss, counter, stop = early_stopping_trigger(val_loss, min_loss, patience, counter)
    if stop:
        print('Early stopping triggered due to consistent increase in validation loss.')
        print("Epoch", epoch-counter, "until", epoch)
        break