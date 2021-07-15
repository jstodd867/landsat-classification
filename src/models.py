import numpy as np
class Baseline():
    
    def __init__(self):
        self.classes_ = None
        self.classes_inverse_ = None
        self.means_ = None
        self.mean_dict = None
        self.label_dict = None
    
    def predict(self, X):
        predicts = []
        self.label_dict = {num:label for num,label in zip(np.arange(len(self.classes_)), self.classes_)}
        for sample in X:
            mean_index = np.argmin([np.sqrt(sum((sample - mean_vector)**2)) for mean_vector in self.mean_dict.values()])
            predicts.append(self.label_dict[mean_index])
        return np.array(predicts)
    
    def fit(self, X_train, y_train):
        self.classes_, self.classes_inverse_ = np.unique(y_train, return_inverse=True)
        self.means_ = [np.mean(X_train[self.classes_inverse_==idx,:], axis=0) for idx,label in enumerate(self.classes_)]
        self.mean_dict = {label: self.means_[i] for i,label in enumerate(self.classes_)}
        