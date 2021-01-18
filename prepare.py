import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from hiragana.image import make_curve_list, make_src, make_trg

def main(data_size, src_path, trg_path):
    curves = [make_curve_list() for _ in range(data_size)]
    p = Pool(10)
    src_list = p.map(make_src, tqdm(curves))
    trg_list = p.map(make_trg, tqdm(curves))
    src_array = np.stack(src_list)
    trg_array = np.stack(trg_list)
    np.save(src_path, src_array)
    np.save(trg_path, trg_array)

if __name__ == '__main__':
    main(2000000, 'data/pretrain.train.src', 'data/pretrain.train.trg')
    main(100, 'data/pretrain.test.src', 'data/pretrain.test.trg')

