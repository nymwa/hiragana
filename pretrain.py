import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from hiragana.dataset import Dataset
from hiragana.autoencoder import AutoEncoder

def make_loader(batch_size):
    src = np.load('data/pretrain.train.src.npy')
    trg = np.load('data/pretrain.train.trg.npy')
    dataset = Dataset(src, trg)
    loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn = dataset.collate)
    return loader

def make_model():
    model = AutoEncoder()
    model = model.cuda()
    print(sum(p.numel() for p in model.parameters()))
    return model

def main():
    clip_norm = 1.0
    max_epochs = 3
    model = make_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), 0.01)
    loader = make_loader(400)
    model.train()
    num_steps = 0
    for epoch in range(max_epochs):
        accum = 0.0
        examples = 0
        for step, (src, trg) in enumerate(tqdm(loader, leave=False)):
            src = src.cuda()
            trg = trg.cuda()
            pred = model(src)
            loss = criterion(pred.flatten(), trg.flatten())
            accum += loss.item() * len(src)
            examples += len(src)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            num_steps += 1
        print('epoch {}: loss {}, steps {}'.format(epoch, accum / examples, num_steps))
        torch.save(model.state_dict(), 'checkpoints/pretrain{}.pt'.format(epoch))

if __name__ == '__main__':
    main()

