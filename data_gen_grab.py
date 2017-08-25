'''
this script makes directories full of 2d imgs
'''
import glob
import nibabel as nib
import pandas
import numpy as np

df = pandas.read_csv('targets.csv')
mapping = {a:b for a,b in zip(df["ID"], df["LABEL_2"]) if not np.isnan(b)}

# define training and testing subjects
portion = 3
full_list = glob.glob('5*.nii.gz')
all_fnames = [f[:5] for f in full_list]
np.random.shuffle(all_fnames)
ratio = len(all_fnames) / portion
train = all_fnames[round(ratio):]
test = all_fnames[:round(ratio)]

READ_INX = 0

def get_label_from_id(id):
    ##TODO
    ### GET LABEL FOR ID IN HERE
    return None


def infinite_slice_generator_with_score(flist, batch_size):
    '''
    ##TODO
    ### READ BATCHES FROM TRAIN LIST STARTING FROM READ_IDX
    ### MODIFY READ_IDX AFTERWARDS.
    '''
    while True: 
        for f in flist:
            score = mapping.get(f)
            if score in [1,-1]:
                N = nib.load(fn).get_data()
                for layer in N:
                    yield layer.shape, score

g = infinite_slice_generator_with_score()
for _ in range(100):
    print(next(g))