import pickle
from typing import List
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


class Retrieval:
    def __init__(self, repository_data: np.array):
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.pca = pickle.load(open(Path(root_path, "pca.pkl"), "rb"))
        self.knn = pickle.load(open(Path(root_path, 'knn.pkl'), 'rb'))

    def inference(self, X: np.array) -> np.array:
        """
        Find 5 images that are most similar to the given image in the repository
        Args:
            X:  All the feature vector of the data which needs to be retrieved the similar images. X.shape=[a, 256],
                a is the number of the data that needs to be retrieved.

        Returns:
            A numpy array with shape=[a, 5], where a is the number of the data that needs to be retrieved. It can
            be seen as a matrix with size=ax5, each row of the matrix is the indices of the 5 images that are most
            similar to the given image in the repository.
        """
        X = self.pca.transform(X)
        y_pred = self.knn.kneighbors(X)[1]
        return y_pred
