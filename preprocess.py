import numpy as np
from PIL import Image, ImageEnhance

kanas = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'

def load_img(path):
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((32, 32))
    img = (255 - np.array(img)) / 255
    img = (img - np.mean(img)) / np.std(img) * 0.15 + 0.05
    img = np.clip(img, 0, 1)
    return img

def main():
    for mode, img_id_list in [('train', [1, 2, 3]), ('valid', [4]), ('test', [5])]:
        src = []
        trg = []
        for img_id in img_id_list:
            for cls_id, char in enumerate(kanas):
                img = load_img('hiraganadata/image{}/{}.png'.format(img_id, char))
                src.append(img)
                trg.append(cls_id)
        src = np.stack(src).astype('float32')
        trg = np.array(trg).astype('int64')
        np.save('data/finetune.{}.src'.format(mode), src)
        np.save('data/finetune.{}.trg'.format(mode), trg)

if __name__ == '__main__':
    main()

