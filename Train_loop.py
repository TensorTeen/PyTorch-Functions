import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, LongT5EncoderModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import time
from tqdm import tqdm


def train(net, train_dataloader, valid_dataloader, criterion, optimizer, epochs=10, device='cpu', checkpoint_epochs=5):
    start = time.time()
    net.to(device)
    print(f'Training for {epochs} epochs on {device}')
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        net.train()  # put network in train mode for Dropout and Batch Normalization
        train_loss = torch.tensor(0., device=device)  # loss and accuracy tensors are on the GPU to avoid data transfers
        train_accuracy = torch.tensor(0., device=device)
        #         N = 0
        for X, y in tqdm(train_dataloader):
            y = y.to(device)
            tokenized = tokenizer.batch_encode_plus(X, return_tensors="pt", padding=True).to(device)
            preds = net(**tokenized)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_accuracy += (torch.argmax(preds, dim=1) == y).sum()
        #                 N += len(X)

        if valid_dataloader is not None:
            net.eval()  # put network in train mode for Dropout and Batch Normalization
            valid_loss = torch.tensor(0., device=device)
            valid_accuracy = torch.tensor(0., device=device)
            with torch.no_grad():
                for X, y in valid_dataloader:
                    y = y.to(device)
                    tokenized = tokenizer.batch_encode_plus(X, return_tensors="pt", padding=True).to(device)
                    preds = net(**tokenized)
                    loss = criterion(preds, y)
                    valid_loss += loss * valid_dataloader.batch_size
                    valid_accuracy += (torch.argmax(preds, dim=1) == y).sum()

        print(f'Training loss: {train_loss / len(train_dataloader.dataset):.2f}')
        print(f'Training accuracy: {100 * train_accuracy / len(train_dataloader.dataset):.2f}')

        if valid_dataloader is not None:
            print(f'Valid loss: {valid_loss / len(valid_dataloader.dataset):.2f}')
            print(f'Valid accuracy: {100 * valid_accuracy / len(valid_dataloader.dataset):.2f}')

        if epoch % checkpoint_epochs == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'./nlp_checkpoint_epoch_{epoch}.pth.tar')

        print()

    end = time.time()
    print(f'Total training time: {end - start:.1f} seconds')
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, './final.pth.tar')
    return net