

import tensorflow as tf
import numpy as np
import h5py
from sklearn.utils import class_weight

class Generator(tf.keras.utils.Sequence):

    def __init__(self, dataset_file, isTrain, config):
        dataset_file = h5py.File(dataset_file,'r')
        self.X = dataset_file['X_train' if isTrain else 'X_val'][:,:,:,:,:config.NUM_CHANNELS]
        self.y = dataset_file['y_train' if isTrain else 'y_val'][:]
        if len(self.X) == 0 or len(self.y) == 0:
            raise ValueError("The generator received empty data.")
        if config.ONLY_NONNULL_INPUTS:
            valid_samples = np.argwhere(np.all(np.all(self.X==0, axis=(1,2,3))==False,axis=-1))[:,0]
            self.X = self.X[valid_samples]
            self.y = self.y[valid_samples]
        self.numSamples = len(self.X)
        self.config = config
        classes_in_y = np.unique(self.y)
        self.class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes_in_y,
            y=self.y.flatten()
        )
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.reshape(np.random.permutation(np.arange(self.numSamples))[:len(self)*self.config.BATCH_SIZE], (len(self),self.config.BATCH_SIZE))
    
    def __getitem__(self, index):
        """
        Fetch a batch by index, apply data augmentation, and return (X, y).
        """
        X = self.X[self.indices[index]]
        y = self.y[self.indices[index]]
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError(f"Empty batch generated at index {index} The generator received empty data.")
        X, y = self.dataAugmentation(X, y)
        return X, y
    
    def __len__(self):
        """
        Number of batches per epoch (floor division).
        """
        return self.numSamples // self.config.BATCH_SIZE
    
    def dataAugmentation(self, X, y):
        """
        Randomly crop both inputs and labels with the same offsets.
        The offsets are sampled per-sample for the (Y, X) plane.
        """
        cropOffsets = np.random.randint(0, high=self.config.CROP_OFFSET, size=(self.config.BATCH_SIZE,2))
        return self.crop(X, cropOffsets), self.crop(y, cropOffsets) if y is not None else None
    
    def crop(self, batch, cropOffsets):
        if len(batch.shape) == 5:
            cropped = np.zeros((self.config.BATCH_SIZE,self.config.ADJACENT_SLICES*2+1,self.config.IMG_SIZE_PADDED,self.config.IMG_SIZE_PADDED,self.config.NUM_CHANNELS))
            for sample in range(self.config.BATCH_SIZE):
                cropped[sample] = batch[sample,:,cropOffsets[sample,0]:cropOffsets[sample,0]+self.config.IMG_SIZE_PADDED,cropOffsets[sample,1]:cropOffsets[sample,1]+self.config.IMG_SIZE_PADDED]
            return cropped
        else:
            cropped = np.zeros((self.config.BATCH_SIZE,self.config.IMG_SIZE,self.config.IMG_SIZE))
            for sample in range(self.config.BATCH_SIZE):
                cropOffsets + cropOffsets + self.config.ADJACENT_SLICES
                cropped[sample] = batch[sample,cropOffsets[sample,0]:cropOffsets[sample,0]+self.config.IMG_SIZE,cropOffsets[sample,1]:cropOffsets[sample,1]+self.config.IMG_SIZE]
            return cropped
    
    def sample_weights(self, y):
        """
        Map per-pixel labels to class weights, producing a weight mask
        with the same shape as y.
        """
        weights = np.zeros(y.shape)
        for i in range(len(self.class_weights)):
            weights[y==i] = self.class_weights[i]
        return weights