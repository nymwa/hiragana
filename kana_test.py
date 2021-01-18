import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hiragana.autoencoder import AutoEncoder
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    src = np.load('data/finetune.train.src.npy')
    model = AutoEncoder()
    model.load_state_dict(torch.load(args.path, map_location='cpu'))
    model.eval()

    for img in src:
        with torch.no_grad():
            img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        pred = model(img)
        img = (img > 0.5).detach().numpy().astype('int8')[0,0]
        hyp = (pred > 0.5).detach().numpy().astype('int8')[0,0]
        for row in np.hstack([img, hyp]):
            print(' '.join([str(x) for x in row]))

if __name__ == '__main__':
    main()

