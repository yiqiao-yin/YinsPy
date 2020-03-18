class YinsMM:

    print("-----------------------------------------------------")
    print(
        """
        Yin's Money Managment Package 
        Copyright © YINS CAPITAL, 2009 – Present
        For more information, please go to www.YinsCapital.com
        """ )
    print("-----------------------------------------------------")
    
    # Define function
    def MarkowitzPortfolio(tickers, start_date, end_date, verbose=True):
        if verbose:
            print("------------------------------------------------------------------------------")
            print("MANUAL: ")
            print("Try run the following line by line in a Python Notebook.")
            print(
                """
                # Load
                %run "../scripts/YinsMM.py"

                # Input
                start_date = pd.to_datetime('2013-01-01')
                end_date = pd.to_datetime('2019-12-6')
                tickers = ['aapl', 'fb'] # only two tickers

                # Run
                temp = YinsMM.MarkowitzPortfolio(tickers, start_date, end_date, verbose=True)
                print('Optimal Portfolio has the following information', testresult['Optimal Portfolio'])
                """ )
            print("Manual ends here.")
            print("------------------------------------------------------------------------------")

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function

        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Verbose
        for i in range(len(tickers)):
            print(
                'Normalize Return:', stockData[tickers[i]]['Normalize Return'], 
                '\n',
                'Expected Return:', np.mean(stockData[tickers[i]]['Normalize Return']),
                '\n',
                'Risk', np.std(stockData[tickers[i]]['Normalize Return']),
                '\n',
                'Sharpe Ratio:', np.mean(stockData[tickers[1]]['Normalize Return']) / np.std(stockData[tickers[0]]['Normalize Return']))
        retMatrix = pd.concat([stockData[tickers[0]]['Normalize Return'], stockData[tickers[1]]['Normalize Return']], axis=1, join='inner')

        # Compute the following for Markowitz Portfolio
        w1 = np.linspace(start=0, stop=1, num=50)
        w2 = 1 - w1
        r1 = np.mean(stockData[tickers[0]]['Normalize Return'])
        r2 = np.mean(stockData[tickers[1]]['Normalize Return'])
        sd1 = np.std(stockData[tickers[0]]['Normalize Return'])
        sd2 = np.std(stockData[tickers[1]]['Normalize Return'])
        rho = np.array(retMatrix.corr())[0][1]

        # Compute paths for returns and risks
        returnPath = np.zeros([1, len(w1)])
        riskPath = np.zeros([1, len(w2)])
        for i in range(len(w1)):
            returnPath[0][i] = w1[i] * r1 + w2[i] * r2
            riskPath[0][i] = w1[i]**2 * sd1**2 + w2[i]**2 * sd2**2 + 2*w1[i]*w2[i]*sd1*sd2*rho

        # Optimal Portfolio
        maximumSR = returnPath / riskPath
        maxVal = maximumSR.max()
        for i in range(len(maximumSR[0])):
            if maximumSR[0][i] == maxVal:
                idx = i

        # Visualization
        import matplotlib.pyplot as plt
        marginsize = 1e-5
        data_for_plot = pd.concat({'Return': pd.DataFrame(returnPath), 'Risk': pd.DataFrame(riskPath)}, axis=0).T
        data_for_plot
        data_for_plot.plot(x='Risk', y='Return', kind='scatter', figsize=(15,5))
        plt.plot(riskPath[0][idx], returnPath[0][idx], marker='o', markersize=10, color='green') # insert an additional dot: this is the position optimal portfolio
        plt.xlim([np.min(riskPath) - marginsize, np.max(riskPath) + marginsize])
        plt.ylim([np.min(returnPath) - marginsize, np.max(returnPath) + marginsize])
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')

        # Output
        return {'Return Matrix': retMatrix, 
                'Correlation Matrix': retMatrix.corr(), 
                'Covariance Matrix': retMatrix.cov(), 
                'Parameters': [w1, w2, r1, r2, sd1, sd2, rho], 
                'Return Path': returnPath, 
                'Risk Path': riskPath,
                'Optimal Portfolio': {
                    'Optimal Weight': [w1[idx], w2[idx]], 
                    'Optimal Return': w1[idx] * r1 + w2[idx] * r2, 
                    'Optimal Volatility': w1[idx]**2 * sd1**2 + w2[idx]**2 * sd2**2 + 2*w1[idx]*w2[idx]*sd1*sd2*rho,
                    'Optimal SR': (w1[idx] * r1 + w2[idx] * r2) / (w1[idx]**2 * sd1**2 + w2[idx]**2 * sd2**2 + 2*w1[idx]*w2[idx]*sd1*sd2*rho)
                }}
    # End of function

    
    # Define function
    def YinsTimer(
        start_date, end_date, ticker, figsize=(15,6), LB=-0.01, UB=0.01, 
        plotGraph=True, verbose=True, printManual=True, gotoSEC=True):
        if printManual:
            print("------------------------------------------------------------------------------")
            print("MANUAL: ")
            print("Try run the following line by line in a Python Notebook.")
            print(
            """
            # Load
            %run "../scripts/YinsMM.py"

            # Run
            start_date = '2010-01-01'
            end_date   = '2020-01-18'
            ticker = 'FB'
            temp = YinsMM.YinsTimer(
                    start_date, end_date, ticker, figsize=(15,6), LB=-0.01, UB=0.01, 
                    plotGraph=True, verbose=True, printManual=True, gotoSEC=True)
            """ )
            print("Manual ends here.")
            print("------------------------------------------------------------------------------")
        
        # Initiate Environment
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import time
                
        # Time
        start = time.time()
        
        # Get Data
        dta = yf.download(ticker, start_date, end_date)
        dta_stock = pd.DataFrame(dta)

        # Define Checking Functions:
        if LB > 0:
            print('Lower Bound (LB) for Signal is not in threshold and is set to default value: -0.01')
            LB = -0.01
        if UB < 0:
            print('Upper Bound (UB) for Signal is not in threshold and is set to default value: +0.01')
            UB = +0.01
        def chk(row):
            if row['aveDIST'] < LB or row['aveDIST'] > UB:
                val = row['aveDIST']
            else:
                val = 0
            return val

        # Generate Data
        df_stock = dta_stock
        close = df_stock['Adj Close']
        df_stock['Normalize Return'] = close / close.shift() - 1

        # Generate Signal:
        if len(dta_stock) < 200:
            data_for_plot = []
            basicStats = []
            print('Stock went IPO within a year.')
        else:
            # Create Features
            df_stock['SMA12'] = close.rolling(window=12).mean()
            df_stock['SMA20'] = close.rolling(window=20).mean()
            df_stock['SMA50'] = close.rolling(window=50).mean()
            df_stock['SMA100'] = close.rolling(window=100).mean()
            df_stock['SMA200'] = close.rolling(window=200).mean()
            df_stock['DIST12'] = close / df_stock['SMA12'] - 1
            df_stock['DIST20'] = close / df_stock['SMA20'] - 1
            df_stock['DIST50'] = close / df_stock['SMA50'] - 1
            df_stock['DIST100'] = close / df_stock['SMA100'] - 1
            df_stock['DIST200'] = close / df_stock['SMA200'] - 1
            df_stock['aveDIST'] = (df_stock['DIST12'] + df_stock['DIST20'] + 
                                   df_stock['DIST50'] + df_stock['DIST100'] + df_stock['DIST200'])/5
            df_stock['Signal'] = df_stock.apply(chk, axis = 1)

            # Plot
            import matplotlib.pyplot as plt
            if plotGraph:
                # No. 1: the first time-series graph plots adjusted closing price and multiple moving averages
                data_for_plot = df_stock[['Adj Close', 'SMA12', 'SMA20', 'SMA50', 'SMA100', 'SMA200']]
                data_for_plot.plot(figsize = figsize)
                plt.show()
                # No. 2: the second time-series graph plots signals generated from investigating distance matrix
                data_for_plot = df_stock[['Signal']]
                data_for_plot.plot(figsize = figsize)
                plt.show()
        
            # Check Statistics:
            SIGNAL      = df_stock['Signal']
            LENGTH      = len(SIGNAL)
            count_plus  = 0
            count_minus = 0
            for i in range(LENGTH):
                if float(SIGNAL.iloc[i,]) > 0:
                    count_plus += 1
            for i in range(LENGTH):
                if float(SIGNAL.iloc[i,]) < 0:
                    count_minus += 1
            basicStats = {'AVE_BUY': round(np.sum(count_minus)/LENGTH, 4),
                          'AVE_SELL': round(np.sum(count_plus)/LENGTH, 4) }

            # Print
            if verbose:
                print("----------------------------------------------------------------------------------------------------")
                print(f"Entered Stock has the following information:")
                print(f'Ticker: {ticker}')
                print("---")
                print(f"Expted Return: {round(np.mean(dta_stock['Normalize Return']), 4)}")
                print(f"Expted Risk (Volatility): {round(np.std(dta_stock['Normalize Return']), 4)}")
                print(f"Reward-Risk Ratio (Daily Data): {round(np.mean(dta_stock['Normalize Return']) / np.std(dta_stock['Normalize Return']), 4)}")
                print("---")
                print("Tail of the 'Buy/Sell Signal' dataframe:")
                print(pd.DataFrame(data_for_plot).tail())
                print("Note: positive values indicate 'sell' and negative values indicate 'buy'.")
                print("---")
                print(f"Basic Statistics for Buy Sell Signals: {basicStats}")
                print("Note: Change LB and UB to ensure average buy sell signals fall beneath 2%.")
                print("---")
                url_front = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
                url_back = "&type=10-K&dateb=&owner=exclude&count=40"
                url_all = str(url_front + ticker + url_back)
                print("For annual report on SEC site, please go to: ")
                print(url_all)
                if gotoSEC:
                    import webbrowser
                    webbrowser.open(url_all)
                print("----------------------------------------------------------------------------------------------------")
        
        # Get More Data:
        tck = yf.Ticker(ticker)
        ALL_DATA = {
            'get stock info': tck.info,
            'get historical market data': tck.history(period="max"),
            'show actions (dividends, splits)': tck.actions,
            'show dividends': tck.dividends,
            'show splits': tck.splits,
            'show financials': [tck.financials, tck.quarterly_financials],
            'show balance sheet': [tck.balance_sheet, tck.quarterly_balance_sheet],
            'show cashflow': [tck.cashflow, tck.quarterly_cashflow],
            'show earnings': [tck.earnings, tck.quarterly_earnings],
            'show sustainability': tck.sustainability,
            'show analysts recommendations': tck.recommendations,
            'show next event (earnings, etc)': tck.calendar
        }
        
        # Time
        end = time.time()
        if verbose == True: 
            print('Time Consumption (in sec):', round(end - start, 2))
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round((end - start)/60/60, 2))

        # Return
        return {'data': dta_stock, 
                'resulting matrix': data_for_plot,
                'basic statistics': basicStats,
                'estimatedReturn': np.mean(dta_stock['Normalize Return']), 
                'estimatedRisk': np.std(dta_stock['Normalize Return']),
                'ALL_DATA': ALL_DATA
               }
    # End function

    
    # Define function
    def CAPM(tickers, start_date, end_date, verbose=True):
        if verbose:
            print("------------------------------------------------------------------------------")
            print("MANUAL: ")
            print("Try run the following line by line in a Python Notebook.")
            print(
                """
                # Load
                %run "../scripts/YinsMM.py"

                # Run
                start_date = pd.to_datetime('2013-01-01')
                end_date = pd.to_datetime('2019-12-6')
                tickers = ['AAPL', 'SPY']
                testresult = CAPM(tickers, start_date, end_date)
                print(testresult['Beta'], testresult['Alpha'])
                """ )
            print("Manual ends here.")
            print("------------------------------------------------------------------------------")
            
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function

        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        import matplotlib.pyplot as plt
        target = stockData[tickers[0]]
        benchmark = stockData[tickers[1]]
        target['Cumulative'] = target['Close'] / target['Close'].iloc[0]
        benchmark['Cumulative'] = benchmark['Close'] / benchmark['Close'].iloc[0]
        target['Cumulative'].plot(label=tickers[0], figsize = (15,5))
        benchmark['Cumulative'].plot(label='Benchmark')
        plt.legend()
        plt.title('Cumulative Return')
        plt.show()

        target['Daily Return'] = target['Close'].pct_change(20)
        benchmark['Daily Return'] = benchmark['Close'].pct_change(20)
        plt.scatter(target['Daily Return'], benchmark['Daily Return'], alpha = 0.3)
        plt.xlabel('Target Returns')
        plt.ylabel('Benchmark Returns')
        plt.title('Daily Returns for Target and Benchmark')
        plt.show()

        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        x = np.array(benchmark['Daily Return']).reshape(-1, 1)
        x = np.array(x[~np.isnan(x)]).reshape(-1, 1)
        y = np.array(target['Daily Return']).reshape(-1, 1)
        y = np.array(y[~np.isnan(y)]).reshape(-1, 1)
        linearModel = lm.fit(x, y)
        y_pred = linearModel.predict(x)

        plt.scatter(x, y, alpha=0.3)
        plt.plot(x, y_pred, 'g')
        plt.xlabel('Benchmark')
        plt.ylabel('Target')
        plt.title('Scatter Dots: Actual Target Returns vs. \nLinear (green): Estimated Target Returns')
        plt.show()

        from sklearn.metrics import r2_score
        score = r2_score(y, y_pred)
        print('R-square is:', score)
        RMSE = np.sqrt(np.mean((y - y_pred)**2))
        print('Root Mean Square Error (RMSE):', RMSE)

        return {'Beta': linearModel.coef_, 
                'Alpha': linearModel.intercept_, 
                'Returns': y, 
                'Estimated Returns': y_pred, 
                'R square': score, 
                'Root Mean Square Error': RMSE}
    # End of function
    
    # Define Function
    def RNN3_Regressor(
        start_date =   '2013-01-01', end_date   =   '2019-12-6',
        tickers    =   'AAPL',       cutoff     =   0.8,
        l1_units   =   50,           l2_units   =   50,           l3_units   =   50,
        optimizer  =   'adam',       loss       =   'mean_squared_error',
        epochs     =   50,           batch_size =   64,
        plotGraph  =   True,         verbose   =   True ):
        
        if verbose:
            print("------------------------------------------------------------------------------")
            print(
                """
                MANUAL: Try run the following line by line in a Python Notebook

                # Load
                %run "../scripts/YinsDL.py"

                # Run
                tmp = RNN3_Regressor(
                        start_date =   '2013-01-01', end_date   =   '2019-12-6',
                        tickers    =   'AAPL',       cutoff     =   0.8,
                        l1_units   =   50,           l2_units   =   50,           l3_units   =   50,
                        optimizer  =   'adam',       loss       =   'mean_squared_error',
                        epochs     =   50,           batch_size =   64,
                        plotGraph  =   True,         verbose   =   True )
                """ )
            print("------------------------------------------------------------------------------")
        
        # Initiate Environment
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt
        import time

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function
        
        start_date = pd.to_datetime(start_date)
        end_date   = pd.to_datetime(end_date)
        tickers    = [tickers]
        
        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Take a look
        # print(stockData[tickers[0]].head(2)) # this is desired stock
        # print(stockData[tickers[1]].head(2)) # this is benchmark (in this case, it is S&P 500 SPDR Index Fund: SPY)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler

        stockData[tickers[0]].iloc[:, 4].head(3)

        data = stockData[tickers[0]].iloc[:, 4:5].values
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_dta = sc.fit_transform(data)
        scaled_dta = pd.DataFrame(scaled_dta)

        training_set = scaled_dta.iloc[0:round(scaled_dta.shape[0] * cutoff), :]
        testing_set = scaled_dta.iloc[round(cutoff * scaled_dta.shape[0] + 1):scaled_dta.shape[0], :]

        # print(training_set.shape, testing_set.shape)

        X_train = []
        y_train = []

        for i in range(100, training_set.shape[0]):
            X_train.append(np.array(training_set)[i-100:i, 0])
            y_train.append(np.array(training_set)[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        if verbose:
            print('--------------------------------------------------------------------')
            print('Shape for data frame in training set:')
            print('Shape of X:', X_train.shape, '; Shape of Y:', len(y_train))
            print('--------------------------------------------------------------------')

        X_test = []
        y_test = []

        for i in range(100, testing_set.shape[0]):
            X_test.append(np.array(testing_set)[i-100:i, 0])
            y_test.append(np.array(testing_set)[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if verbose:
            print('--------------------------------------------------------------------')
            print('Shape for data frame in testing set:')
            print('Shape of X:', X_test.shape, ': Shape of Y:', len(y_test))
            print('--------------------------------------------------------------------')

        ### Build RNN
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        import time

        # Initialize RNN
        begintime = time.time()
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units = 1))
        endtime = time.time()

        # Summary
        if verbose:
            print("--------------------------------------------")
            print('Let us investigate the sequential models.')
            regressor.summary()
            print("--------------------------------------------")
            print("Time Consumption (in sec):", endtime - begintime)
            print("Time Consumption (in min):", round((endtime - begintime)/60, 2))
            print("Time Consumption (in hr):", round((endtime - begintime)/60)/60, 2)
            print("--------------------------------------------")

        ### Train RNN
        # Compiling the RNN
        start = time.time()
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        end = time.time()
        
        # Time Check
        if verbose == True: 
            print('Time Consumption:', end - start)

        ### Predictions
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        real_stock_price = np.reshape(y_test, (y_test.shape[0], 1))
        real_stock_price = sc.inverse_transform(real_stock_price)

        ### Performance Visualization

        # Visualising the results
        import matplotlib.pyplot as plt
        if plotGraph:
            plt.plot(real_stock_price, color = 'red', label = f'Real {tickers[0]} Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {tickers[0]} Stock Price')
            plt.title(f'{tickers[0]} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(f'{tickers[0]} Stock Price')
            plt.legend()
            plt.show()

        import math
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        if verbose:
            print(f'---------------------------------------------------------------------------------')
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'------------------')
            print(f'Interpretation:')
            print(f'------------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')
            print(f'---------------------------------------------------------------------------------')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function
    
    # Define Function
    def RNN4_Regressor(
        start_date = '2013-01-01',
        end_date   = '2019-12-6',
        tickers    = 'AAPL', cutoff = 0.8,
        l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
        optimizer = 'adam', loss = 'mean_squared_error',
        epochs = 50, batch_size = 64,
        plotGraph = True,
        verbose = True
    ):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Run
        tmp = YinsDL.RNN4_Regressor(
            start_date = '2013-01-01',
            end_date   = '2019-12-6',
            tickers    = 'FB', cutoff = 0.8,
            l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
            optimizer = 'adam', loss = 'mean_squared_error',
            epochs = 30, batch_size = 64,
            plotGraph = True,
            verbose = True )
        """
        
        # Initiate Environment
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt
        import time
        
        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function
        
        start_date = pd.to_datetime(start_date)
        end_date   = pd.to_datetime(end_date)
        tickers    = [tickers]
        
        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Take a look
        # print(stockData[tickers[0]].head(2)) # this is desired stock
        # print(stockData[tickers[1]].head(2)) # this is benchmark (in this case, it is S&P 500 SPDR Index Fund: SPY)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler

        stockData[tickers[0]].iloc[:, 4].head(3)

        data = stockData[tickers[0]].iloc[:, 4:5].values
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_dta = sc.fit_transform(data)
        scaled_dta = pd.DataFrame(scaled_dta)

        training_set = scaled_dta.iloc[0:round(scaled_dta.shape[0] * cutoff), :]
        testing_set = scaled_dta.iloc[round(cutoff * scaled_dta.shape[0] + 1):scaled_dta.shape[0], :]

        # print(training_set.shape, testing_set.shape)

        X_train = []
        y_train = []

        for i in range(100, training_set.shape[0]):
            X_train.append(np.array(training_set)[i-100:i, 0])
            y_train.append(np.array(training_set)[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        if verbose:
            print('--------------------------------------------------------------------')
            print('Shape for data frame in training set:')
            print('Shape of X:', X_train.shape, '; Shape of Y:', len(y_train))
            print('--------------------------------------------------------------------')

        X_test = []
        y_test = []

        for i in range(100, testing_set.shape[0]):
            X_test.append(np.array(testing_set)[i-100:i, 0])
            y_test.append(np.array(testing_set)[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if verbose:
            print('--------------------------------------------------------------------')
            print('Shape for data frame in testing set:')
            print('Shape of X:', X_test.shape, ': Shape of Y:', len(y_test))
            print('--------------------------------------------------------------------')

        ### Build RNN
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialize RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l4_units))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units = 1))

        if verbose:
            print('Let us investigate the summary of the sequential models.')
            regressor.summary()

        ### Train RNN
        # Compiling the RNN
        start = time.time()
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        end = time.time()
        
        # Time Check
        if verbose == True: 
            print("---------------------------------------------------")
            print('Time Consumption (in sec):', end - start)
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round(((end - start)/60)/60), 2)
            print("---------------------------------------------------")

        ### Predictions
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        real_stock_price = np.reshape(y_test, (y_test.shape[0], 1))
        real_stock_price = sc.inverse_transform(real_stock_price)

        ### Performance Visualization
        # Visualising the results
        import matplotlib.pyplot as plt
        if plotGraph:
            plt.plot(real_stock_price, color = 'red', label = f'Real {tickers[0]} Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {tickers[0]} Stock Price')
            plt.title(f'{tickers[0]} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(f'{tickers[0]} Stock Price')
            plt.legend()
            plt.show()

        import math
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        if verbose:
            print(f'---------------------------------------------------------------------------------')
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'------------------')
            print(f'Interpretation:')
            print(f'------------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')
            print(f'---------------------------------------------------------------------------------')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function
    
    # Define Function
    def RNN10_Regressor(
        start_date = '2013-01-01',
        end_date   = '2019-12-6',
        tickers    = 'AAPL', cutoff = 0.8,
        l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
        l5_units = 50, l6_units = 50, l7_units = 50, l8_units = 50,
        l9_units = 50, l10_units = 50,
        optimizer = 'adam', loss = 'mean_squared_error',
        epochs = 50, batch_size = 64,
        plotGraph = True,
        verbose = True
    ):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Run
        tmp = YinsMM.RNN4_Regressor(
            start_date = '2013-01-01',
            end_date   = '2019-12-6',
            tickers    = 'FB', cutoff = 0.8,
            l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
            l5_units = 50, l6_units = 50, l7_units = 50, l8_units = 50,
            l9_units = 50, l10_units = 50,
            optimizer = 'adam', loss = 'mean_squared_error',
            epochs = 30, batch_size = 64,
            plotGraph = True,
            verbose = True )
        """
        
        # Initiate Environment
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt
        import time

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function
        
        start_date = pd.to_datetime(start_date)
        end_date   = pd.to_datetime(end_date)
        tickers    = [tickers]
        
        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Take a look
        # print(stockData[tickers[0]].head(2)) # this is desired stock
        # print(stockData[tickers[1]].head(2)) # this is benchmark (in this case, it is S&P 500 SPDR Index Fund: SPY)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler

        stockData[tickers[0]].iloc[:, 4].head(3)

        data = stockData[tickers[0]].iloc[:, 4:5].values
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_dta = sc.fit_transform(data)
        scaled_dta = pd.DataFrame(scaled_dta)

        training_set = scaled_dta.iloc[0:round(scaled_dta.shape[0] * cutoff), :]
        testing_set = scaled_dta.iloc[round(cutoff * scaled_dta.shape[0] + 1):scaled_dta.shape[0], :]

        # print(training_set.shape, testing_set.shape)

        X_train = []
        y_train = []

        for i in range(100, training_set.shape[0]):
            X_train.append(np.array(training_set)[i-100:i, 0])
            y_train.append(np.array(training_set)[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        if verbose:
            print('--------------------------------------------------------------------')
            print('Shape for data frame in training set:')
            print('Shape of X:', X_train.shape, '; Shape of Y:', len(y_train))
            print('--------------------------------------------------------------------')

        X_test = []
        y_test = []

        for i in range(100, testing_set.shape[0]):
            X_test.append(np.array(testing_set)[i-100:i, 0])
            y_test.append(np.array(testing_set)[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if verbose:
            print('--------------------------------------------------------------------')
            print('Shape for data frame in testing set:')
            print('Shape of X:', X_test.shape, ': Shape of Y:', len(y_test))
            print('--------------------------------------------------------------------')

        ### Build RNN
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialize RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l4_units, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a fifth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l5_units, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a sixth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l6_units, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a seventh LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l7_units, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a eighth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l8_units, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a nighth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l9_units, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a tenth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l10_units))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units = 1))

        if verbose:
            print('Let us investigate the summary of the sequential models.')
            regressor.summary()

        ### Train RNN
        # Compiling the RNN
        start = time.time()
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
        end = time.time()
        
        # Time Check
        if verbose == True: 
            print('Time Consumption:', end - start)

        ### Predictions
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        real_stock_price = np.reshape(y_test, (y_test.shape[0], 1))
        real_stock_price = sc.inverse_transform(real_stock_price)

        ### Performance Visualization
        # Visualising the results
        import matplotlib.pyplot as plt
        if plotGraph:
            plt.plot(real_stock_price, color = 'red', label = f'Real {tickers[0]} Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {tickers[0]} Stock Price')
            plt.title(f'{tickers[0]} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(f'{tickers[0]} Stock Price')
            plt.legend()
            plt.show()

        import math
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        if verbose:
            print(f'---------------------------------------------------------------------------------')
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'------------------')
            print(f'Interpretation:')
            print(f'------------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')
            print(f'---------------------------------------------------------------------------------')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function
    
        
    # Define function
    def YinsInvestigator(
        start_date, end_date, ticker, figsize=(15,6), LB=-0.01, UB=0.01, pastNdays=10,
        plotGraph=True, verbose=True, printManual=True, printSummary=True, gotoSEC=True, showInteractive=True):
        if printManual:
            print("------------------------------------------------------------------------------")
            print("MANUAL: ")
            print("Try run the following line by line in a Python Notebook.")
            print(
            f"""
            # Load
            %run "../scripts/YinsMM.py"

            # Run
            start_date = '2010-01-01'
            end_date   = '2020-01-18'
            ticker = '{ticker}'
            temp = YinsMM.YinsTimer(
                    start_date, end_date, ticker, figsize=(15,6), LB=-0.01, UB=0.01, 
                    plotGraph=True, verbose=True, printManual=True, gotoSEC=True)
            """ )
            print("Manual ends here.")
            print("------------------------------------------------------------------------------")
        
        # Initiate Environment
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import time
        import hvplot.pandas
        import plotly.express as px
                
        # Time
        start = time.time()
        
        # Get Data
        dta = yf.download(ticker, start_date, end_date)
        dta_stock = pd.DataFrame(dta)
        
        # Get More Data:
        tck = yf.Ticker(ticker)
        ALL_DATA = {
            'get stock info': tck.info,
            'get historical market data': tck.history(period="max"),
            'show actions (dividends, splits)': tck.actions,
            'show dividends': tck.dividends,
            'show splits': tck.splits,
            'show financials': [tck.financials, tck.quarterly_financials],
            'show balance sheet': [tck.balance_sheet, tck.quarterly_balance_sheet],
            'show cashflow': [tck.cashflow, tck.quarterly_cashflow],
            'show earnings': [tck.earnings, tck.quarterly_earnings],
            'show sustainability': tck.sustainability,
            'show analysts recommendations': tck.recommendations,
            'show next event (earnings, etc)': tck.calendar }

        # Define Checking Functions:
        if LB > 0:
            print('Lower Bound (LB) for Signal is not in threshold and is set to default value: -0.01')
            LB = -0.01
        if UB < 0:
            print('Upper Bound (UB) for Signal is not in threshold and is set to default value: +0.01')
            UB = +0.01
        def chk(row):
            if row['aveDIST'] < LB or row['aveDIST'] > UB:
                val = row['aveDIST']
            else:
                val = 0
            return val

        # Generate Data
        df_stock = dta_stock
        close = df_stock['Adj Close']
        df_stock['Normalize Return'] = close / close.shift() - 1

        # Generate Signal:
        if len(dta_stock) < 200:
            data_for_plot = []
            basicStats = []
            print('Stock went IPO within a year.')
        else:
            # Create Features
            df_stock['SMA12'] = close.rolling(window=12).mean()
            df_stock['SMA20'] = close.rolling(window=20).mean()
            df_stock['SMA50'] = close.rolling(window=50).mean()
            df_stock['SMA100'] = close.rolling(window=100).mean()
            df_stock['SMA200'] = close.rolling(window=200).mean()
            df_stock['DIST12'] = close / df_stock['SMA12'] - 1
            df_stock['DIST20'] = close / df_stock['SMA20'] - 1
            df_stock['DIST50'] = close / df_stock['SMA50'] - 1
            df_stock['DIST100'] = close / df_stock['SMA100'] - 1
            df_stock['DIST200'] = close / df_stock['SMA200'] - 1
            df_stock['aveDIST'] = (df_stock['DIST12'] + df_stock['DIST20'] + 
                                   df_stock['DIST50'] + df_stock['DIST100'] + df_stock['DIST200'])/5
            df_stock['Signal'] = df_stock.apply(chk, axis = 1)

            # Plot
            import matplotlib.pyplot as plt
            if plotGraph:
                # No. 1: the first time-series graph plots adjusted closing price and multiple moving averages
                data_for_plot_chart = df_stock[['Adj Close', 'SMA12', 'SMA20', 'SMA50', 'SMA100', 'SMA200']]
                data_for_plot_chart.plot(figsize = figsize)
                plt.show()
                # No. 2: the second time-series graph plots signals generated from investigating distance matrix
                data_for_plot_signal = df_stock[['Signal', 'aveDIST']]
                data_for_plot_signal.plot(figsize = figsize)
                plt.show()
            if showInteractive:
                chart_hvplot  = data_for_plot_chart.hvplot(ylabel='Price (in USD)', alpha=0.7)
                signal_hvplot = data_for_plot_signal.hvplot(
                    title='Average Distance from Price to Moving Averages & Signals (by LB and UB)', alpha=0.7)
                data_for_parallel = df_stock[['Adj Close', 'Signal', 'aveDIST']]
                chart_signal_plotly = px.parallel_coordinates(data_for_parallel, color='Signal')
        
            # Check Statistics:
            SIGNAL      = df_stock['Signal']
            LENGTH      = len(SIGNAL)
            count_plus  = 0
            count_minus = 0
            for i in range(LENGTH):
                if float(SIGNAL.iloc[i,]) > 0:
                    count_plus += 1
            for i in range(LENGTH):
                if float(SIGNAL.iloc[i,]) < 0:
                    count_minus += 1
            basicStats = {'AVE_BUY': round(np.sum(count_minus)/LENGTH, 4),
                          'AVE_SELL': round(np.sum(count_plus)/LENGTH, 4) }
            # Print
            if verbose:
                print("----------------------------------------------------------------------------------------------------")
                print(f"Entered Stock has the following information:")
                print(f'Ticker: {ticker}')
                print("---")
                if printSummary:
                    print(f"Business Summary: {ALL_DATA['get stock info']['longBusinessSummary']}")
                print("---")
                print(f"Expted Return: {round(np.mean(dta_stock['Normalize Return']), 4)}")
                print(f"Expted Risk (Volatility): {round(np.std(dta_stock['Normalize Return']), 4)}")
                print(f"Reward-Risk Ratio (Daily Data): {round(np.mean(dta_stock['Normalize Return']) / np.std(dta_stock['Normalize Return']), 4)}")
                print("---")
                print("Tail of the 'Buy/Sell Signal' dataframe:")
                print(pd.DataFrame(data_for_plot_signal).tail(pastNdays))
                print("Note: ")
                print("- positive values indicate 'sell' and negative values indicate 'buy'.")
                print("- source: Go to https://yinscapital.com/research/ and click on 'Buy Signal from Limit Theorem'.")
                print("---")
                print(f"Basic Statistics for Buy Sell Signals: {basicStats}")
                print("Note: Change LB and UB to ensure average buy sell signals fall beneath 2%.")
                print("---")
                sec_url_front = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
                sec_url_back = "&type=10-K&dateb=&owner=exclude&count=40"
                sec_url_all = str(sec_url_front + ticker + sec_url_back)
                print("For annual report on SEC site, please go to: ")
                print(sec_url_all)
                url_front_finviz = "https://finviz.com/quote.ashx?t="
                url_finviz = str(url_front_finviz + ticker)
                print("For news and charts on FinViz.com, please go to: ")
                print(url_finviz)
                if gotoSEC:
                    import webbrowser
                    webbrowser.open(sec_url_all)
                    webbrowser.open(url_finviz)
                print("----------------------------------------------------------------------------------------------------")
                
        # Time
        end = time.time()
        if verbose == True: 
            print('Time Consumption (in sec):', round(end - start, 2))
            print('Time Consumption (in min):', round((end - start)/60, 2))
            print('Time Consumption (in hr):', round((end - start)/60/60, 2))

        # Return
        return {'data': dta_stock, 
                'updated data': df_stock,
                'resulting matrix': [data_for_plot_chart, data_for_plot_signal],
                'basic statistics': basicStats,
                'estimatedReturn': np.mean(dta_stock['Normalize Return']), 
                'estimatedRisk': np.std(dta_stock['Normalize Return']),
                'ALL_DATA': ALL_DATA,
                'InterPlot': {'chart': chart_hvplot, 'signal': signal_hvplot, 'paraPlot': chart_signal_plotly}}