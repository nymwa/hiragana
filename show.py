import numpy as np
import torch

def main():
    src = np.load('data/test.src.npy')
    trg = np.load('data/test.trg.npy')
    np.set_printoptions(threshold=10000)
    for i in range(100):
        x = trg[i]
        x = x > 0.1
        x = x.astype('int')
        print(x)

if __name__ == '__main__':
    main()

