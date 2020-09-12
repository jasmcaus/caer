# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

import h5py 

class HDF5Dataset:
    def __init__(self, shape, dataset_name, buffer_size=1000):
        if '.h5' not in dataset_name:
            dataset_name += '.h5'
        
        # Creating 2 datasets -> Features/Images and corresponding labels
        self.db = h5py.File('dataset_name', 'w')

        self.features = self.db.create_dataset(name='features', shape=shape, dtype='float')
        self.labels = self.db.create_dataset(name='labels', shape=shape[0], dtype='int')

        # # We aren't adding data directly because that is handled via the buffer
        # self.features = self.db.create_dataset(name='features', shape=shape, data=X)
        # self.labels = self.db.create_dataset(name='labels', shape=shape[0], data=y)

        # Storing buffer size and initializing the buffer
        self.buffSize = buffer_size
        self.buffer = {'features': [], 'labels': []}
        self.buff_idx = 0


def create_dataset(X, y, dataset_name):
    """
    Creates an h5 dataset of features and corresponding labels
    :param X: feature set
    :param y: labels
    :param dataset_name: Name for the dataset
    """

    if '.h5' not in dataset_name:
        dataset_name += '.h5'

    h5f = h5py.File('dataset_name', 'w')
    h5f.create_dataset('features', data=X)
    h5f.create_dataset('labels', data=y)
    h5f.close()
    

def load_dataset(dataset_name):
    if '.h5' not in dataset_name:
        dataset_name = dataset_name + '.h5'
    
    h5f = h5py.File('dataset_name', 'r')
    X = h5f['features'][:] 
    y = h5f['labels'][:]
    h5f.close()

    return X, y