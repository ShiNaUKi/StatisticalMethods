import numpy as np
import matplotlib.pyplot as plt

# Section 1: Load package
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4,cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],
                    y=X[y==cl,1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')

    # visualizin the test set
    if test_idx:
        X_test,y_test=X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='yellow',edgecolor='black',alpha=1.0,
                    linewidth=1,marker='o',s=100,label='test set')



# 1 simple Perceptron
# 2. Perceptron from scikit-learn



class Perceptron_demo:
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
    # 1.simple Perceptron
    # X = np.array([[1,1], [2,2], [3,4]])
    # Y0 = np.array([-1, 1, 1,])
    #
    # model = Perceptron_demo(n_fea=2)
    # model.fit_offline(X, Y0)
    #
    #
    # plt.scatter(X[:1,0],X[:1, 1] , color='r')
    # plt.scatter(X[1:, 0], X[1:, 1], color='g')
    #
    # plt.plot(np.arange(0, 6, 0.01), model.calc_y(np.arange(0, 6, 0.01)), color='b')
    # plt.show()
    #
    # print(f"w = {model.w}, b = {model.b}")

    # 2. Perceptron from scikit-learn
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    print(f'statistical of labels y = {np.bincount(y)}')

    # Preprocessing
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    # Training
    ppn = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=1)
    ppn.fit(X_train_scaled, y_train)
    y_pred = ppn.predict(X_test_scaled)
    print(f'misclassification rate is {np.sum(y_test != y_pred) / 100}')
    print(f'acc1 {accuracy_score(y_test, y_pred)}')
    print(f'acc2 {ppn.score(X_test_scaled, y_test)}')

    # Visualize
    X_for_vis = preprocessing.scale(X[:, :2])
    Y_for_vis = y

    plot_decision_regions(X=X_for_vis,
                          y=Y_for_vis,
                          classifier=ppn,
                          test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


