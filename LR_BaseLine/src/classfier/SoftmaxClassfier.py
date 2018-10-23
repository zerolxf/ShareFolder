from numpy import *
from Classfier import *
from model import *


class SoftmaxClassfier(Classfier):
    def __init__(self, _w, _model):
        super().__init__(_w)
        self.model = _model

    def predict(self, data_x):
        pred = self.model.softmax(self.w.T*data_x)
        num = -1
        prob = 0
        for i in range(pred.shape[0]):
            if pred[i][0] > prob:
                num = i
                prob = pred[i][0]
        return num, prob
