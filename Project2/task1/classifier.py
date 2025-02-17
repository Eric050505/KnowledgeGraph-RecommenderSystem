import numpy as np
import pickle
from pathlib import Path
import os


class Classifier:
    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.svm = pickle.load(open(Path(root_path, 'classification_svm_model.pkl'), 'rb'))
        self.scaler = pickle.load(open(Path(root_path, 'classification_scaler.pkl'), 'rb'))

    def inference(self, X: np.array) -> np.array:
        X = self.scaler.transform(X)
        y_pred = self.svm.predict(X)
        return y_pred
