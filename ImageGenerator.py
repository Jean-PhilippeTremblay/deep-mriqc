from scipy import ndimage
import numpy as np
import os
import threading
import warnings
import skimage.transform as transform
import math
import random

class Iterator(object):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if self.x.ndim != 5:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.do_crop(x)
            x = self.image_data_generator.do_resample(x)
            x = self.image_data_generator.do_random_transform(x)
            batch_x[i] = x
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

class ImageGenerator(object):
    def __init__(self,
                 crop_size=None,
                 resample_size=None,
                 normalize_by='one',
                 x_rotation_max_angel_deg=0,
                 y_rotation_max_angel_deg=0,
                 z_rotation_max_angel_deg=0,
                 ):
        self.crop = type(resample_size) == tuple
        self.crop_size = crop_size
        self.normalize_by = normalize_by
        self.resample = type(resample_size) == tuple
        self.resample_size = resample_size
        self.x_rotation_max_angel_deg = x_rotation_max_angel_deg
        self.y_rotation_max_angel_deg = y_rotation_max_angel_deg
        self.z_rotation_max_angel_deg = z_rotation_max_angel_deg

    def flow(self, x, y, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            )

    def do_crop(self, x):
        if self.crop:
            mid_slice_x = round(x.shape[0] / 2)
            start_slice_x = mid_slice_x - round(self.crop_size[0] / 2)
            mid_slice_y = round(x.shape[1] / 2)
            start_slice_y = mid_slice_y - round(self.crop_size[1] / 2)
            mid_slice_z = round(x.shape[2] / 2)
            start_slice_z = mid_slice_z - round(self.crop_size[2] / 2)
            sliced = x[start_slice_x:start_slice_x + self.crop_size[0], start_slice_y:start_slice_y + self.crop_size[1],
                     start_slice_z:start_slice_z + self.crop_size[2]]
            if type(self.normalize_by) == str and self.normalize_by in ['max', 'mean', 'min', 'one']:
                if self.normalize_by == 'max':
                    return sliced / np.max(sliced[:])
                elif self.normalize_by == 'mean':
                    return sliced / np.mean(sliced[:])
                elif self.normalize_by == 'min':
                    return sliced / np.min(sliced[:])
                elif self.normalize_by == 'one':
                    return sliced / 1
            else:
                raise Exception('Normalization param error must be in {0}'.format(['max', 'mean', 'min', 'one']))
        else:
            return x

    def do_resample(self, x):
        if self.resample:
            print(x.shape)
            return transform.resize(x, self.resample_size, preserve_range=True, mode='constant')
        else:
            return x

    def do_random_transform(self, x):
        x_rot_mat = np.eye(3,3)
        if self.x_rotation_max_angel_deg > 0:
            rot_deg = random.randint(-1*self.x_rotation_max_angel_deg,self.x_rotation_max_angel_deg)
            x_rot_mat[1, 1] = math.cos(math.radians(rot_deg))
            x_rot_mat[2, 2] = math.cos(math.radians(rot_deg))
            x_rot_mat[1, 2] = -1 * math.sin(math.radians(rot_deg))
            x_rot_mat[2, 1] = math.sin(math.radians(rot_deg))

        y_rot_mat = np.eye(3, 3)
        if self.y_rotation_max_angel_deg > 0:
            rot_deg = random.randint(-1 * self.y_rotation_max_angel_deg, self.y_rotation_max_angel_deg)
            y_rot_mat[0, 0] = math.cos(math.radians(rot_deg))
            y_rot_mat[2, 2] = math.cos(math.radians(rot_deg))
            y_rot_mat[2, 0] = -1 * math.sin(math.radians(rot_deg))
            y_rot_mat[0, 2] = math.sin(math.radians(rot_deg))

        z_rot_mat = np.eye(3, 3)
        if self.z_rotation_max_angel_deg > 0:
            rot_deg = random.randint(-1 * self.z_rotation_max_angel_deg, self.z_rotation_max_angel_deg)
            z_rot_mat[0, 0] = math.cos(math.radians(rot_deg))
            z_rot_mat[1, 1] = math.cos(math.radians(rot_deg))
            z_rot_mat[0, 1] = -1 * math.sin(math.radians(rot_deg))
            z_rot_mat[1, 0] = math.sin(math.radians(rot_deg))

        full_rot_mat = np.dot(np.dot(x_rot_mat, y_rot_mat), z_rot_mat)
        return ndimage.affine_transform(x, full_rot_mat)
