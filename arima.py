import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic', 'p-value',
              'Number of Lags Used', 'Number of Observations Used']

    for value, label in zip(result, labels):
        print(label+' : '+str(value))

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \n")


def AR(p, df, key):
    '''
    Generating the lagged p terms
    '''
    df_temp = df

    for i in range(1, p+1):
        df_temp['Shifted_values_%d' % i] = df_temp[key].shift(i)

    train_size = (int)(0.8 * df_temp.shape[0])

    # Breaking data set into test and training
    df_train = pd.DataFrame(df_temp[0:train_size])
    df_val = pd.DataFrame(df_temp[train_size:df_temp.shape[0]])

    df_train_2 = df_train.dropna()
    # X contains the lagged values ,hence we skip the first column
    try:
        X_train = df_train_2.iloc[:, 1:].values.reshape(-1, p)
    except:
        return [pd.DataFrame(), pd.DataFrame(), 0, 0, float("inf")]
    # Y contains the value,it is the first column
    y_train = df_train_2.iloc[:, 1].values.reshape(-1, 1)

    # Running linear regression to generate the coefficents of lagged terms
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

    theta = np.array(model.params[1:])
    intercept = model.params[0]
    df_train_2['Predicted_Values'] = X_train.dot(theta) + intercept
    # df_train_2[['Value','Predicted_Values']].plot()

    X_val = df_val.iloc[:, 1:].values.reshape(-1, p)
    df_val['Predicted_Values'] = X_val.dot(theta) + intercept
    # df_test[['Value','Predicted_Values']].plot()

    RMSE = np.sqrt(mean_squared_error(
        df_val[key], df_val['Predicted_Values']))
    AIC = model.aic
    BIC = model.bic

    print("The RMSE is :", RMSE, ", Value of p : ",
          p, ", AIC is : ", AIC, ", BIC is : ", BIC)
    return [df_train_2, df_val, theta, intercept, RMSE, AIC, BIC]


def MA(q, res):
    ''' Moving Average'''

    for i in range(1, q+1):
        res['Shifted_values_%d' % i] = res['Residuals'].shift(i)

    train_size = (int)(0.8 * res.shape[0])

    res_train = pd.DataFrame(res[0:train_size])
    res_val = pd.DataFrame(res[train_size:res.shape[0]])

    res_train_2 = res_train.dropna()

    X_train = res_train_2.iloc[:, 1:].values.reshape(-1, q)
    y_train = res_train_2.iloc[:, 0].values.reshape(-1, 1)

    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

    theta = np.array(model.params[1:])
    intercept = model.params[0]

    res_train_2['Predicted_Values'] = X_train.dot(theta) + intercept
    # res_train_2[['Residuals','Predicted_Values']].plot()

    X_val = res_val.iloc[:, 1:].values.reshape(-1, q)
    res_val['Predicted_Values'] = X_val.dot(theta) + intercept
    res_val[['Residuals', 'Predicted_Values']].plot()

    from sklearn.metrics import mean_squared_error
    RMSE = np.sqrt(mean_squared_error(
        res_val['Residuals'], res_val['Predicted_Values']))
    AIC = model.aic
    BIC = model.bic

    print("The RMSE is:", RMSE, ", Value of AIC is:", AIC,
          ", Value of BIC is:", BIC, ", Value of q:", q)
    return [res_train_2, res_val, theta, intercept, RMSE, AIC, BIC]


def opt_p(df_testing, low_p, up_p, str, key):
    ''' Pick the best p based on lowest aic or bic or RMSE as specified'''
    assert (str == 'AIC' or str == 'BIC' or str == 'RMSE')
    best_p_aic, best_p_bic, best_p_rmse = 0, 0, 0
    best_rmse, best_aic, best_bic = float("inf"), float("inf"), float("inf")
    for p in range(low_p, up_p):
        df_train_2, df_val, theta, intercept, RMSE, AIC, BIC = AR(
            p, pd.DataFrame(df_testing), key)
        if RMSE < best_rmse:
            best_p_rmse = p
            best_rmse = RMSE
        if AIC < best_aic:
            best_p_aic = p
            best_aic = AIC
        if BIC < best_bic:
            best_p_bic = p
            best_bic = BIC
    if (str == 'RMSE'):
        return best_p_rmse, best_rmse
    elif (str == 'AIC'):
        return best_p_aic, best_aic
    elif (str == 'BIC'):
        return best_p_bic, best_bic


def opt_q(df_testing, low_q, up_q, str):
    ''' Pick the best q based on lowest aic or bic or RMSE as specified'''
    assert (str == 'AIC' or str == 'BIC' or str == 'RMSE')
    best_q_aic, best_q_bic, best_q_rmse = 0, 0, 0
    best_rmse, best_aic, best_bic = float("inf"), float("inf"), float("inf")
    for q in range(low_q, up_q):
        res_train_2, res_val, theta, intercept, RMSE, AIC, BIC = MA(
            q, pd.DataFrame(df_testing))
        if RMSE < best_rmse:
            best_q_rmse = q
            best_rmse = RMSE
        if AIC < best_aic:
            best_q_aic = q
            best_aic = AIC
        if BIC < best_bic:
            best_q_bic = q
            best_bic = BIC
    if (str == 'RMSE'):
        return best_q_rmse, best_rmse
    elif (str == 'AIC'):
        return best_q_aic, best_aic
    elif (str == 'BIC'):
        return best_q_bic, best_bic


def arima(low_p, high_p, low_q, high_q, df, key):
    # Call AR
    copy = df.copy(deep=True)
    # Can have 'BIC', 'AIC' or 'RMSE' as the last argument
    best_p, best_error = opt_p(copy, low_p, high_p, 'AIC', key)
    df_train, df_test, AR_theta, AR_intercept, AR_RMSE, AR_AIC, AR_BIC = AR(
        best_p, pd.DataFrame(df[key]), key)

    # Combined dataframe results from AR (? -- I'm unsure ?)
    df_c = pd.concat([df_train, df_test])
    # Calculate residuals
    res = pd.DataFrame()
    res['Residuals'] = df_c[key] - df_c.Predicted_Values

    # Call MA
    copy = res.copy(deep=True)
    # Can have 'BIC', 'AIC' or 'RMSE' as the last argument
    best_q, best_error = opt_q(copy, low_q, high_q, 'AIC')
    res_train, res_val, MA_theta, MA_intercept, MA_RMSE, MA_AIC, MA_BIC = MA(
        best_q, res)

    # Combined residuals dataframe
    res_c = pd.concat([res_train, res_val])

    df_c.Predicted_Values += res_c.Predicted_Values

    return df_c, AR_theta, AR_intercept, AR_RMSE, MA_theta, MA_intercept, MA_RMSE, AR_AIC, AR_BIC, MA_AIC, MA_BIC


df = pd.read_csv('data/pars_normal_daily.csv')
print(arima(1, 21, 1, 21, pd.DataFrame(df.mu), 'mu'))
