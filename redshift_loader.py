import scipy
import numpy as np
from keras.utils import Sequence
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py


class DataLoader(Sequence):
    def __init__(self, dataset_name, batch_size, img_res=(101, 101), norm=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.img_res = img_res
        self.data = np.loadtxt("network/networkFrame.csv", delimiter=',', dtype=str)
        self.h5 = h5py.File("network/0-4000.hdf5", mode='r')
        self.norm = norm

    def __getitem__(self, item):
        batch_images = self.data[(item*self.batch_size):((item+1)*self.batch_size)]

        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def on_epoch_end(self):
        return NotImplementedError

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
        if object_id < 4000:
            redshift = self.h5[object_id]["z_source"]



    def load_data(self, batch_size=1, is_testing=False):
        batch_images = np.random.choice(self.data.shape[0], size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img_A, img_B = self.imread(self.data[img_path])

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            if not is_testing and np.random.random() > 0.5:
                img_A = np.flipud(img_A)
                img_B = np.flipud(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.asarray(imgs_A)
        imgs_B = np.asarray(imgs_B)
        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):

        self.n_batches = int(len(self.data) / batch_size)

        for i in range(self.n_batches - 1):
            batch = self.data[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                # Need to load in 5 channels here for the data
                img_A, img_B = self.imread(img)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                if not is_testing and np.random.random() < 0.5:
                    img_A = np.flipud(img_A)
                    img_B = np.flipud(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.asarray(imgs_A)
            imgs_B = np.asarray(imgs_B)
            yield imgs_A, imgs_B

    def imread(self, path):
        sim_img = []
        source_img = []
        for element in path[1:]:
            if "sci" in element:
                if "sim" in element:
                    img = fits.getdata(element, ext=0)
                    center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
                    img = img[center[0] - 32:center[0] + 32, center[1] - 32:center[1] + 32]
                    sim_img.append(img)
                elif "source" in element:
                    img = fits.getdata(element, ext=0)
                    center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
                    img = img[center[0] - 32:center[0] + 32, center[1] - 32:center[1] + 32]
                    source_img.append(img)
        sim_img = np.asarray(sim_img).T
        source_img = np.asarray(source_img).T
        if self.norm:
            sim_img = 2 * (sim_img - np.min(sim_img)) / (np.max(sim_img) - np.min(sim_img)) - 1.
            source_img = 2 * (source_img - np.min(source_img)) / (np.max(source_img) - np.min(source_img)) - 1.

        return sim_img, source_img

