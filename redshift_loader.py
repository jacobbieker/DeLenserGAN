import scipy
import numpy as np
from keras.utils import Sequence
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.utils import shuffle
import h5py


class DataLoader(Sequence):
    def __init__(self, batch_size, img_res=(101, 101), norm=False, data=None):
        self.batch_size = batch_size
        self.img_res = img_res
        if data is None:
            self.data = np.loadtxt("network/networkFrame.csv", delimiter=',', dtype=str)
        else:
            self.data = data
        self.h5 = h5py.File("network/0-4000.hdf5", mode='r')
        self.h52 = h5py.File("network/4000-7137.hdf5", mode='r')
        self.norm = norm

    def __getitem__(self, item):
        batch_images = self.data[(item*self.batch_size):((item+1)*self.batch_size)]
        return self.load_batch(batch_images)

    def __len__(self):
        return int(len(self.data)/self.batch_size)

    def on_epoch_end(self):
        self.data = shuffle(self.data)


    def load_image(self, path):
        sim_img = []
        for element in path[1:]:
            if "sci" in element:
                if "sim" in element:
                    img = fits.getdata(element, ext=0)
                    center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
                    img = img[center[0] - 32:center[0] + 32, center[1] - 32:center[1] + 32]
                    sim_img.append(img)
        sim_img = np.asarray(sim_img).T
        if self.norm:
            sim_img = 2 * (sim_img - np.min(sim_img)) / (np.max(sim_img) - np.min(sim_img)) - 1.

        return sim_img

    def load_redshift(self, object_id):
        object_id = object_id - 1
        if object_id < 4000:
            data = self.h5
            redshift = data["z_source"][object_id]
        else:
            data = self.h52
            redshift = data["z_source"][int(object_id - 4000)]
        return redshift

    def load_batch(self, paths):
        batch = paths
        imgs_A = []
        redshifts = []
        for img in batch:
            # Need to load in 5 channels here for the data
            img_A = self.load_image(img)
            redshift = self.load_redshift(int(img[0]))

            if np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
            if np.random.random() < 0.5:
                img_A = np.flipud(img_A)

            imgs_A.append(img_A)
            redshifts.append(redshift)
        imgs_A = np.asarray(imgs_A)
        redshifts = np.asarray(redshifts)
        return imgs_A, redshifts

