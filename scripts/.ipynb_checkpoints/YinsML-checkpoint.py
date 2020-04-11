class YinsML:

    """
    Yin's Machine Learning Package 
    Copyright © YINS CAPITAL, 2009 – Present
    """

    # Define function
    def DecisionTree_Classifier(X_train, X_test, y_train, y_test, maxdepth = 3):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree
        
        # Train
        DCT = tree.DecisionTreeClassifier(max_depth=maxdepth)
        DCT = DCT.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = DCT.predict(X_train)
        y_train_hat_score = DCT.predict_proba(X_train)

        from sklearn.metrics import confusion_matrix
        confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
        confusion_train
        
        train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
        train_acc

        y_test_hat_ = DCT.predict(X_test)
        y_test_hat_score = DCT.predict_proba(X_test)
        confusion_test = pd.DataFrame(confusion_matrix(y_test_hat_, y_test))
        confusion_test

        test_acc = sum(np.diag(confusion_test)) / sum(sum(np.array(confusion_test)))
        test_acc
        
        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': DCT,
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'y_train_hat_score': y_train_hat_score,
                'confusion_train': confusion_train,
                'train_acc': train_acc
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'y_test_hat_score': y_test_hat_score,
                'confusion_test': confusion_test,
                'test_acc': test_acc
            }
        }
    # End of function
    
    # Define function
    def DecisionTree_Regressor(X_train, X_test, y_train, y_test, maxdepth = 3):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree
        
        # Train
        DCT = tree.DecisionTreeClassifier(max_depth=maxdepth)
        DCT = DCT.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = DCT.predict(X_train)
        RMSE_train = np.sqrt(np.mean((y_train_hat_ - y_train)**2))

        # Report Out-of-sample Estimators
        y_test_hat_ = DCT.predict(X_test)
        RMSE_test = np.sqrt(np.mean((y_test_hat_ - y_test)**2))
        
        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': DCT,
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'RMSE_train': RMSE_train
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'RMSE_test': RMSE_test
            }
        }
    # End of function
    
    # Define function
    def adam_optimizer(Xadam, y, batch_size = 10, lr = 0.01, epochs = 200, period = 20, verbose=True):
        
        # Library
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # Adam
        def adam(params, vs, sqrs, lr, batch_size, t):
            beta1 = 0.1
            beta2 = 0.111
            eps_stable = 1e-9

            for param, v, sqr in zip(params, vs, sqrs):
                g = param.grad / batch_size

                v[:] = beta1 * v + (1. - beta1) * g
                sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

                v_bias_corr = v / (1. - beta1 ** t)
                sqr_bias_corr = sqr / (1. - beta2 ** t)

                div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
                param[:] = param - div

        # Library
        import mxnet as mx
        from mxnet import autograd
        from mxnet import ndarray as nd
        from mxnet import gluon
        import random

        mx.random.seed(1)
        random.seed(1)

        # Generate data.
        # Xadam = covid19_confirmed_china_rolling_data.iloc[:, [1,2,3,5]] <=== this is input
        num_inputs = pd.DataFrame(Xadam).shape[1]
        num_examples = pd.DataFrame(Xadam).shape[0]
        X = nd.array(Xadam)
        # y = nd.array(covid19_confirmed_china_rolling_data['Y']) <=== this is input
        dataset = gluon.data.ArrayDataset(X, y)

        # Construct data iterator.
        def data_iter(batch_size):
            idx = list(range(num_examples))
            random.shuffle(idx)
            for batch_i, i in enumerate(range(0, num_examples, batch_size)):
                j = nd.array(idx[i: min(i + batch_size, num_examples)])
                yield batch_i, X.take(j), y.take(j)

        # Initialize model parameters.
        def init_params():
            w = nd.random_normal(scale=1, shape=(num_inputs, 1))
            b = nd.zeros(shape=(1,))
            params = [w, b]
            vs = []
            sqrs = []
            for param in params:
                param.attach_grad()
                vs.append(param.zeros_like())
                sqrs.append(param.zeros_like())
            return params, vs, sqrs

        # Linear regression.
        def net(X, w, b):
            return nd.dot(X, w) + b

        # Loss function.
        def square_loss(yhat, y):
            return (yhat - y.reshape(yhat.shape)) ** 2 / 2

        # %matplotlib inline
        import matplotlib as mpl
        mpl.rcParams['figure.dpi']= 120
        import matplotlib.pyplot as plt
        import numpy as np

        def train(batch_size, lr, epochs, period):
            assert period >= batch_size and period % batch_size == 0
            [w, b], vs, sqrs = init_params()
            total_loss = [np.mean(square_loss(net(X, w, b), y).asnumpy())]

            t = 0
            # Epoch starts from 1.
            for epoch in range(1, epochs + 1):
                for batch_i, data, label in data_iter(batch_size):
                    with autograd.record():
                        output = net(data, w, b)
                        loss = square_loss(output, label)
                    loss.backward()
                    # Increment t before invoking adam.
                    t += 1
                    adam([w, b], vs, sqrs, lr, batch_size, t)
                    if batch_i * batch_size % period == 0:
                        total_loss.append(np.mean(square_loss(net(X, w, b), y).asnumpy()))
                print("Batch size %d, Learning rate %f, Epoch %d =========================> loss %.4e" %
                      (batch_size, lr, epoch, total_loss[-1]))
            print('w:', np.reshape(w.asnumpy(), (1, -1)),
                  'b:', b.asnumpy()[0], '\n')
            x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
            plt.semilogy(x_axis, total_loss)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()

            return w, b

        w, b = train(batch_size = batch_size, lr = lr, epochs = epochs, period = period)

        w_adam = []
        for w_i in range(len(list(w.asnumpy()))):
            w_adam.append(list(w.asnumpy())[w_i][0])
        if verbose: 
            print('Weight:', w_adam)

        b_adam = list(b.asnumpy())[0]
        if verbose:
            print('Bias:', b_adam)

        y_hat_adam = np.dot(Xadam, w_adam) + b_adam

        return {
            'parameters': {'w': w, 'b': b},
            'y_estimate': y_hat_adam
        }
    # End of function