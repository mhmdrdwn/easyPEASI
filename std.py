from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler

"""Standardization of 3D matrix is taken from the stackoverflow post
https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix"""

class StandardScaler3D(BaseEstimator,TransformerMixin):
    #batch, sequence, channels
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self,X,y=None):
        self.scaler.fit(X.reshape(-1, X.shape[2]))
        return self

    def transform(self,X):
        return self.scaler.transform(X.reshape( -1,X.shape[2])).reshape(X.shape)

def standardize(train_set, valid_set, test_set):
    scaler=StandardScaler3D()
    trainset_X=np.moveaxis(np.array(train_set.X),1,2)
    validset_X=np.moveaxis(np.array(valid_set.X),1,2)
    testset_X=np.moveaxis(np.array(test_set.X),1,2)

    trainset_X=scaler.fit_transform(trainset_X)
    validset_X=scaler.transform(validset_X)
    testset_X=scaler.transform(testset_X)

    trainset_X=np.moveaxis(trainset_X,1,2)
    validset_X=np.moveaxis(validset_X,1,2)
    testset_X=np.moveaxis(testset_X,1,2)
    return trainset_X, validset_X, testset_X
