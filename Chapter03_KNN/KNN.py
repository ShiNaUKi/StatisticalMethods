import numpy as np

# 1. sample KNN
# 2. KNN kdTree
# 3. KNN and kdTree from code libraries

class KNN_classifier:
    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None

    def fit(self, x, y):
        if self.x is None:
            self.x = x
        else:
            self.x = np.concatenate(self.x, x)

        if self.y is None:
            self.y = y
        else:
            self.y = np.concatenate(self.y, y)

    def predict(self, x):
        preds = []

        if len(x.shape) <= 1:
            x = x.reshape((1, -1))

        for fea in x:
            dist = np.sum((fea-self.x)**2, axis=1)
            index = np.argsort(dist)[:self.k]
            y_labels = self.y[index]
            pred = np.argmax(np.bincount(y_labels))
            preds.append(pred)
        return preds

class KNN_td_tree:
    def __init__(self):
        pass



if __name__ == "__main__":
    X = np.array([[2,3],
                  [4,7],
                  [5,4],
                  [7,2],
                  [9,6],
                  [8,1]])
    Y = np.array([0, 0, 0, 1, 1, 1])

    sample = np.array([[4, 4], [8, 1], [10, 2]])

    model = KNN_classifier(k=3)
    model.fit(X, Y)
    pred = model.predict(sample)
    print(pred)
