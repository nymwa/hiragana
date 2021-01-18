import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from hiragana.autoencoder import AutoEncoder, Classifier
from argparse import ArgumentParser

def load_data(mode):
    src = np.load('data/finetune.{}.src.npy'.format(mode))
    trg = np.load('data/finetune.{}.trg.npy'.format(mode))
    src = torch.tensor(src).unsqueeze(1).cuda()
    trg = torch.tensor(trg).cuda()
    return src, trg

def make_model(path = None):
    model = Classifier()
    if path is not None:
        pretrained = AutoEncoder()
        pretrained.load_state_dict(torch.load(path,  map_location='cpu'))
        model.encoder = pretrained.encoder
    model = model.cuda()
    print(sum(p.numel() for p in model.parameters()))
    return model

def accuracy(pred, trg):
    hyp = torch.softmax(pred, dim=-1).argmax(dim=-1) == trg
    return hyp.sum() / len(hyp)


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--checkpoint', default=None)
    args = parser.parse_args()

    max_epochs = 200
    clip_norm = 0.1
    train_src, train_trg = load_data('train')
    valid_src, valid_trg = load_data('valid')
    test_src, test_trg = load_data('test')
    model = make_model(args.checkpoint)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 0.005)
    transform = transforms.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1))
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(transform(train_src))
        train_loss = criterion(train_pred, train_trg)
        train_loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        train_accuracy = accuracy(train_pred, train_trg)

        model.eval()
        with torch.no_grad():
            valid_pred = model(valid_src)
        valid_loss = criterion(valid_pred, valid_trg)
        valid_accuracy = accuracy(valid_pred, valid_trg)

        model.eval()
        with torch.no_grad():
            test_pred = model(test_src)
        test_loss = criterion(test_pred, test_trg)
        test_accuracy = accuracy(test_pred, test_trg)

        print('epoch {}: train_loss {:.3f}, valid_loss {:.3f}, train_accuracy: {:.3f}, valid_accuracy: {:.3f}, test_accuracy: {:.3f}, total_norm: {:.3f}'.format(
            epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, test_accuracy, total_norm))
        # torch.save(model.state_dict(), 'checkpoints/finetune{}.pt'.format(epoch))

if __name__ == '__main__':
    main()

