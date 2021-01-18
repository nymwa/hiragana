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

    src = np.load('data/pretrain.test.src.npy')
    trg = np.load('data/pretrain.test.trg.npy')
    model = AutoEncoder()
    model.load_state_dict(torch.load(args.path, map_location='cpu'))
    model.eval()

    for img_id in range(100):
        with torch.no_grad():
            img = torch.from_numpy(src[img_id]).float().unsqueeze(0).unsqueeze(0)
        pred = model(img)
        hyp = (torch.sigmoid(pred) > 0.5).detach().numpy().astype('int8')[0,0]
        ref = (trg[img_id] > 0.5).astype('int8')
        acc = sum(hyp.ravel() == ref.ravel()) / len(hyp.ravel())
        print(acc)
        cat = np.hstack([hyp, ref])
        for row in cat:
            print(' '.join([str(x) for x in row]))

if __name__ == '__main__':
    main()

