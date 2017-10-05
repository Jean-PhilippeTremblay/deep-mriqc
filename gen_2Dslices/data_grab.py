from nilearn import image
import numpy, functools
from multiprocessing import Pool

def get_slices_normalised(slices, nif):
    data = nif.get_data()
    mid_slice_x = round(data.shape[0]/2)
    start_slice_x = mid_slice_x - round(slices/2)
    mid_slice_y = round(data.shape[1] / 2)
    start_slice_y = mid_slice_y - round(slices / 2)
    sliced = data[start_slice_x:start_slice_x+slices,start_slice_y:start_slice_y+slices,start_slice_y:start_slice_y+slices]
    max = numpy.max(sliced[:])
    return sliced/max

def resample_img(moving, fixed):
    return image.resample_to_img(moving, fixed)


def get_data(sub_id_lab, data_dir):
    sub_id_lab = sub_id_lab.strip()
    sub_id = sub_id_lab.split('\t')[0]
    label = sub_id_lab.split('\t')[1]
    base_dir = '{0}/deep_abide'.format(data_dir)
    fixed = '50002'
    file_path = '{base}/{file_base}.nii.gz'.format(base=base_dir, file_base=sub_id)
    try:
        sub_img = resample_img(file_path, '{base}/{file_base}.nii.gz'.format(base=base_dir, file_base=fixed))
    except:
        return None, None
    return get_slices_normalised(80, sub_img), label

def get_data_no_crop(sub_id_lab, data_dir):
    sub_id_lab = sub_id_lab.strip()
    sub_id = sub_id_lab.split('\t')[0]
    label = sub_id_lab.split('\t')[1]
    base_dir = '{0}/deep_abide'.format(data_dir)
    fixed = '50002'
    file_path = '{base}/{file_base}.nii.gz'.format(base=base_dir, file_base=sub_id)
    try:
        sub_img = resample_img(file_path, '{base}/{file_base}.nii.gz'.format(base=base_dir, file_base=fixed))
    except:
        return None, None
    return sub_img.get_data(), label

def get_all_data(sub_list, data_dir):
    pool = Pool()
    f_list =  list(pool.map(functools.partial(get_data, data_dir=data_dir), sub_list))
    data_list = [f[0] for f in f_list]
    lab_list = [f[1] for f in f_list]
    data_list = [d for d in data_list if isinstance(d, numpy.ndarray)]
    lab_list = [int(float(l)) for l in lab_list if l]
    return data_list, lab_list

def get_all_data_no_crop(sub_list, data_dir):
    pool = Pool()
    f_list =  list(pool.map(functools.partial(get_data_no_crop, data_dir=data_dir), sub_list))
    data_list = [f[0] for f in f_list]
    lab_list = [f[1] for f in f_list]
    data_list = [d for d in data_list if isinstance(d, numpy.ndarray)]
    lab_list = [int(float(l)) for l in lab_list if l]
    return data_list, lab_list

def all_data(sc_dir, data_dir):
    ff = open('{0}/rater_2.tsv'.format(sc_dir)).readlines()
    return get_all_data(ff, data_dir)

def all_data_no_crop(sc_dir, data_dir):
    ff = open('{0}/rater_2.tsv'.format(sc_dir)).readlines()
    return get_all_data_no_crop(ff, data_dir)

if __name__ == '__main__':
    all_data()
