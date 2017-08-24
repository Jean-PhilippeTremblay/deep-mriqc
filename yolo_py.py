import glob, random
from nilearn import image
import matplotlib.image as mim
import numpy as np
import pandas as pd

df = pd.read_csv('rater_2.tsv', sep='\t')
mapping = dict(zip(df["ID"].get_values(), df["LABEL_2"].get_values()))

def get_resampled_images(files):
    first, *rest = files
    fixed = image.load_img(first)
    yield fixed, mapping[int(first[:5])]

    for path in rest:
        if int(path[:5]) in mapping:
            yield image.resample_to_img(path, fixed), mapping[int(path[:5])]

def get_block(image, block_height):
    N = image.get_data().astype(float)
    N -= N.min()
    if N.max() > 0: N /= N.max()
    N = (N*256).astype('uint16')
    # ^ normalize into 0 - 256 range
    mid = len(N)//2
    half = block_height//2
    return N[mid-half:mid+half]

files = glob.glob("5*.nii.gz")
layers =  [(layer, score) for img, score in get_resampled_images(files) for layer in get_block(img, 40)]
random.shuffle(layers)
data, scores = zip(*layers)
np.savez('data', np.dstack(data).T)
np.savez('scores', scores)
