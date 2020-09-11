# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

import h5py 

def create_dataset(X, y, dataset_name):
    """
    Creates an h5 dataset of features and corresponding labels
    :param X: feature set
    :param y: labels
    :param dataset_name: Name for the dataset
    """

    if '.h5' not in dataset_name:
        dataset_name = dataset_name + '.h5'

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
