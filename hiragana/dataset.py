import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
    
    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        return {'src': self.src[index],
                'trg': self.trg[index]}
        
    def collate(self, batch):
        src = torch.stack([torch.from_numpy(x['src']).clone() for x in batch]).unsqueeze(1)
        trg = torch.stack([torch.from_numpy(x['trg']).clone() for x in batch]).unsqueeze(1)
        return src, trg

