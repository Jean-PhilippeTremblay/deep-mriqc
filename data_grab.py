from nilearn import image
import numpy

def get_slices_normalised(slices, nif):
    data = nif.get_data()
    mid_slice_n = round(data.shape[0]/2)
    start_slice = mid_slice_n - round(slices/2)
    sliced = data[start_slice:start_slice+slices,:,:]
    max = numpy.max(sliced[:])
    return sliced/max

def resample_img(moving, fixed):
    return image.resample_to_img(moving, fixed)


def get_data(sub_id):
    sub_id = sub_id.strip()
    base_dir = '/dbh_data/deep_abide'
    fixed = '50002'
    file_path = '{base}/{file_base}.nii.gz'.format(base=base_dir, file_base=sub_id)
    sub_img = resample_img(file_path, '{base}/{file_base}.nii.gz'.format(base=base_dir, file_base=fixed))
    return get_slices_normalised(40, sub_img)

def get_all_data(sub_list):
    return list(map(get_data, sub_list))


def all_data():
    ff = open('rater_2_IDs.tsv').readlines()
    return get_all_data(ff)


if __name__ == '__main__':
    all_data()