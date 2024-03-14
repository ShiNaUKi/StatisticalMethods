import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_fea, n=1):
        self.w = np.zeros(n_fea)
        self.b = 1e05
        self.n = n

    def predict(self, x):
        pred = np.sum(self.w * x + self.b)
        if pred == 0:
            return 0
        else:
            return np.sign(np.sum(self.w * x + self.b))

    def calc_y(self, x):
        return -(self.w[1] * x + self.b) / self.w[0]


    def update(self, x, y):
        self.w = self.w + self.n * y * x
        self.b = self.b + self.n * y

    def calc_loss(self, x, y):
        return -np.sum(np.tile(y, (1,len(self.w))).reshape(len(self.w),-1).T * (x * self.w  + self.b))

    def fit_offline(self, x, y):
        pre_loss = self.calc_loss(x, y)
        self.fit(x, y)
        loss = self.calc_loss(x, y)

        while (pre_loss > loss):
            loss = pre_loss
            print(f"updating !!!! pre_loss = {pre_loss}, now_loss = {loss}")
            self.fit(x, y)
            loss = self.calc_loss(x, y)


    def fit(self,  X, y0):
        for x, y in zip(X, y0):
            while (self.predict(x) == 0 or self.predict(x) != y):
                self.update(x, y)
                print(f"{x}, {y} update, w = {self.w}, b = {self.b}")



if __name__ == "__main__":
    X = np.array([[1,1], [2,2], [3,4]])
    Y0 = np.array([-1, 1, 1,])

    model = Perceptron(n_fea=2)
    model.fit_offline(X, Y0)


    plt.scatter(X[:1,0],X[:1, 1] , color='r')
    plt.scatter(X[1:, 0], X[1:, 1], color='g')

    plt.plot(np.arange(0, 6, 0.01), model.calc_y(np.arange(0, 6, 0.01)), color='b')
    plt.show()

    print(f"w = {model.w}, b = {model.b}")