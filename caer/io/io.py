# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus
# pylint: disable=unused-variable

import h5py 

class HDF5Dataset:
    def __init__(self, shape, dataset_name, buffer_size=1000):
        """
            shape holds the dimensions of the data. 
            For the MNIST Dataset, shape = (70000, 28, 28). This can be flattened into a feature vector (to be fed into any Machine Learning classifier like SVM or Logistic Regression) by reshaping to (70000, 784) --> 28*28 = 784
        """
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
    
    def add(self, features, labels):
        # Adding rows and labels to the buffer
        self.buffer['features'].extend(features)
        self.buffer['labels'].extend(labels)

        # If buffer needs to be flushed to disk (if >buffer_size)
        if len(self.buffer['features']) >= self.buffSize:
            self.flush()
    
    def flush(self):
        # Buffers and written to disk and then flushed
        idx = self.buff_idx + len(self.buffer['features'])
        self.features[self.buff_idx : idx] = self.buffer['features']
        self.labels[self.buff_idx : idx] = self.buffer['labels']

        # Resetting buffer
        self.buff_idx = idx
        self.buffer = {'features': [], 'labels': []}
    
    def storeClassLabels(self, classLabels):
        """
            Stores the actual classLabels as an additional piece of the dataset
            classLabels must be a LIST
        """
        dtype = h5py.special_dtype(vlen='unicode')
        classLabels_set = self.db.create_dataset('class labels', shape=len(classLabels), dtype=dtype, data = classLabels)

    def close(self):
        # Writing any remaining items in buffer to disk
        if len(self.buffer['data']) > 0:
            self.flush()
        
        # Closing the dataset
        self.db.close()

# def create_dataset(X, y, dataset_name):
#     """
#     Creates an h5 dataset of features and corresponding labels
#     :param X: feature set
#     :param y: labels
#     :param dataset_name: Name for the dataset
#     """

#     if '.h5' not in dataset_name:
#         dataset_name += '.h5'

#     h5f = h5py.File('dataset_name', 'w')
#     h5f.create_dataset('features', data=X)
#     h5f.create_dataset('labels', data=y)
#     h5f.close()
    

def load_dataset(dataset_name):
    if '.h5' not in dataset_name:
        dataset_name = dataset_name + '.h5'
    
    h5f = h5py.File('dataset_name', 'r')
    X = h5f['features'][:] 
    y = h5f['labels'][:]
    h5f.close()

    return X, y