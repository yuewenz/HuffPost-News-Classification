import numpy as np

class MLR():
    def __init__(self, n_labels):
        self.n_labels = n_labels
        # thetas: num_label * D
        self.thetas = None 

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def addbias(self, x):
        N, D = np.shape(x)

        xtest = np.ones((N, D+1))
        xtest[:, 1:] = x

        return xtest

    def predict(self, x):
        # x: N * D
        # thetas: num_label * D
        # likelihood: N * num_labels
        likelihood = np.dot(x, self.thetas.T)

        return np.argmax(likelihood, axis = 1)

    def fit(self, xtrain, ytrain, learning_rate = 0.01, n_it = 100, tol = 0.01, 
        c_lambda = 1.0, useSGD = False, regularization = False):
        # parameters
        N, D = np.shape(xtrain)
        learning_rate = 10.0#1 * 1e-2
        n_it = 1000
        tol = 1e-2
        c_lambda = 1.0
        self.thetas = np.zeros((self.n_labels, D))

        # stochastic gradient descent
        useSGD = False
        regularization = False
        if useSGD:
            # some feature might not be used?
            Ndata = 1000#int(max(N/100, 100))
            new_xtrain = xtrain[:Ndata]
            new_ytrain = ytrain[:Ndata]
        else:
            Ndata = N
            new_xtrain = xtrain
            new_ytrain = ytrain
        # calculate theta for each label
        # theta[i] is the weights for likelihood being label i
        for i in range(self.n_labels):
            print(i)
            theta0 = np.zeros((D,1))

            for it in range(int(n_it)):
                if useSGD:
                    indices = np.random.choice(N, Ndata, replace = False)
                    new_xtrain = xtrain[indices]
                    new_ytrain = ytrain[indices]

                y = np.zeros((Ndata, 1))
                y[new_ytrain == i] = 1
                loss = y - self.sigmoid(np.dot(new_xtrain, theta0))
                theta = theta0 + learning_rate * np.dot(new_xtrain.T, loss) / Ndata

                # theta = theta0 - learning_rate / Ndata * \
                #         np.sum(new_xtrain[new_ytrain == i,:], axis = 0).reshape((D,1))
                # exp_xtheta = np.exp(-np.dot(new_xtrain, theta0))
                # D*N @ N*1
                # theta += learning_rate / Ndata * \
                #         np.dot(new_xtrain.T, (1 - self.sigmoid(np.dot(new_xtrain, theta0)))) 
                        # np.dot(new_xtrain.T, (exp_xtheta / (1 + exp_xtheta))) 

                if regularization:
                    theta -= c_lambda * theta0 / Ndata

                error = np.linalg.norm(theta - theta0)
                # print(error)
                if error < tol: break
                theta0 = theta

            self.thetas[i,:] = theta.reshape((1,D))
        # print(ytrain[:100])
        # print(self.predict(xtrain[:100]))

        


