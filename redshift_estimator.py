import h5py
import numpy

f = h5py.File("network/0-4000.hdf5", 'r')

print(f.keys())