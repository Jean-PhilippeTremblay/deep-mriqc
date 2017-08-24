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