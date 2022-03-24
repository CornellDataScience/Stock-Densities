import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression


def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','Number of Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \n")


def AR(p, df, key):
    '''
    Generating the lagged p terms
    '''
    df_temp = df

    for i in range(1,p+1):
        df_temp['Shifted_values_%d' % i ] = df_temp[key].shift(i)

    train_size = (int)(0.8 * df_temp.shape[0])

    #Breaking data set into test and training
    df_train = pd.DataFrame(df_temp[0:train_size])
    df_val = pd.DataFrame(df_temp[train_size:df_temp.shape[0]])

    df_train_2 = df_train.dropna()
    #X contains the lagged values ,hence we skip the first column
    try:
        X_train = df_train_2.iloc[:,1:].values.reshape(-1,p)
    except:
        return [pd.DataFrame(), pd.DataFrame(), 0, 0, float("inf")]
    #Y contains the value,it is the first column
    y_train = df_train_2.iloc[:,1].values.reshape(-1,1)

    #Running linear regression to generate the coefficents of lagged terms
    lr = LinearRegression()
    lr.fit(X_train,y_train)

    theta  = lr.coef_.T
    intercept = lr.intercept_
    df_train_2['Predicted_Values'] = X_train.dot(lr.coef_.T) + lr.intercept_
    # df_train_2[['Value','Predicted_Values']].plot()

    X_val = df_val.iloc[:,1:].values.reshape(-1,p)
    df_val['Predicted_Values'] = X_val.dot(lr.coef_.T) + lr.intercept_
    # df_test[['Value','Predicted_Values']].plot()

    RMSE = np.sqrt(mean_squared_error(df_val[key], df_val['Predicted_Values']))

    print("The RMSE is :", RMSE,", Value of p : ",p)
    return [df_train_2,df_val,theta,intercept,RMSE]


def MA(q,res):
    ''' Moving Average'''

    for i in range(1,q+1):
        res['Shifted_values_%d' % i ] = res['Residuals'].shift(i)

    train_size = (int)(0.8 * res.shape[0])

    res_train = pd.DataFrame(res[0:train_size])
    res_val = pd.DataFrame(res[train_size:res.shape[0]])

    res_train_2 = res_train.dropna()

    X_train = res_train_2.iloc[:,1:].values.reshape(-1,q)
    y_train = res_train_2.iloc[:,0].values.reshape(-1,1)

    lr = LinearRegression()
    lr.fit(X_train,y_train)

    theta  = lr.coef_.T
    intercept = lr.intercept_
    res_train_2['Predicted_Values'] = X_train.dot(lr.coef_.T) + lr.intercept_
    # res_train_2[['Residuals','Predicted_Values']].plot()

    X_val = res_val.iloc[:,1:].values.reshape(-1,q)
    res_val['Predicted_Values'] = X_val.dot(lr.coef_.T) + lr.intercept_
    #  res_val[['Residuals','Predicted_Values']].plot()

    RMSE = np.sqrt(mean_squared_error(res_val['Residuals'], res_val['Predicted_Values']))

    print("The RMSE is :", RMSE,", Value of q : ",q)
    return [res_train_2,res_val,theta,intercept,RMSE]


def opt_p (df_testing, low_p, up_p):
    ''' Pick the best p based on cross validation '''
    best_p = 0
    best_rmse = float("inf")
    for p in range(low_p, up_p):
        df_train_2,df_val,theta,intercept,RMSE = AR(p, pd.DataFrame(df_testing))
        if RMSE<best_rmse:
            best_p = p
            best_rmse = RMSE
    return best_p, best_rmse


def opt_q (res, low_q, up_q):
    ''' Pick the best q based on cross validation '''
    best_q = 0
    best_rmse = float("inf")
    for q in range(low_q, up_q):
        res_train_2,res_val,theta,intercept,RMSE = MA(q, pd.DataFrame(res))
        if RMSE<best_rmse:
            best_q = q
            best_rmse = RMSE
    return best_q, best_rmse

def arima(p,q,df,key):
    # Call AR
    df_train, df_test, AR_theta, AR_intercept, AR_RMSE = AR(p, pd.DataFrame(df[key]), key)

    # Combined dataframe results from AR (? -- I'm unsure ?)
    df_c = pd.concat([df_train, df_test])
    # Calculate residuals
    res = pd.DataFrame()
    res['Residuals'] = df_c[key] - df_c.Predicted_Values

    # Call MA
    res_train, res_val, MA_theta, MA_intercept, MA_RMSE = MA(q, res)

    # Combined residuals dataframe
    res_c = pd.concat([res_train, res_val])

    df_c.Predicted_Values += res_c.Predicted_Values

    return df_c, AR_theta, AR_intercept, AR_RMSE, MA_theta, MA_intercept, MA_RMSE
