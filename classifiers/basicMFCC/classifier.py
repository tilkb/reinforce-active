from sklearn import svm
import numpy as np

class CustomClassifier:
    def __init__(self):
        self.clf1 = svm.SVC(probability=True, decision_function_shape='ovo',
                            kernel='rbf', gamma=0.001, C=50)
        self.reset()
        
    def reset(self):
        self.classes = {}
        self.classes_back = {}
        self.one_class=-1
        
        

    def fit(self,X,y):
        self.reset()
        temp=[]
        for item in y:
            if not(item in self.classes):
                self.classes[item]=len(self.classes)
                self.classes_back[self.classes[item]]=item
                self.one_class=item
            temp.append(self.classes[item])
        train_X=np.array(X)
        train_Y=np.array(temp)

        if len(self.classes)>1:
            self.clf1.fit(train_X,train_Y)

    def predict(self,X):
        if len(self.classes)==0:
            return None
        if len(self.classes)==1:
            return np.ones(len(X))*self.one_class
        result = self.clf1.predict(X)
        result_final=[]
        for item in result:
            result_final.append(self.classes_back[item])
        return result_final
        

