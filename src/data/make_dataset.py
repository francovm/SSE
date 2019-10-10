
import requests
import pandas as pd
import datetime
import numpy as np
from sklearn import linear_model

# stations_list = open("/home/francovm/Projects/SSE/data/external/GISB.txt").read().splitlines()
stations_list = open("/home/francovm/Projects/SSE/data/external/hikurangi_gnss.txt").read().splitlines()

def GNSS_dataframe(data):
    """
    This function turns the string of GNSS data received by requests.get
    into a data frame with GNSS data correctly formatted.
    """
    data = data.split("\n") # splits data on the new line symbol
    for i in range(0, len(data)):
        data[i]= data[i].split(",")# splits data ponits on the , symbol
    for i in range(1, (len(data)-1)):
        data[i][0] = datetime.datetime.strptime(data[i][0], '%Y-%m-%dT%H:%M:%S.%fZ') #make 1st value into a datetime object
        data[i][1] = float(data[i][1]) #makes 2nd value into a decimal number
        data[i][2] = float(data[i][2]) #makes 3rd value into a decimal number
    df = pd.DataFrame(data[1:-1],index = range(1, (len(data)-1)), columns=data[0]) #make the list into a data frame
    return df


def Regression_GPS(data, window):
    """
    This function calculate linear regression over the East and North component
    and smooth the slope coefficient in the same window
    """

    reg = linear_model.LinearRegression()
    pendiente = []
    ordenada = []
    data['counter'] = range(len(data))

    for i in range(window, len(data)):
        temp = data.iloc[i - window:i, :]

        counter_temp = np.array(temp['counter'].values.reshape(-1, 1))
        y_e_temp = np.array(temp['x'].values.reshape(-1))

        reg.fit(counter_temp, y_e_temp)

        m_temp = reg.coef_
        b_temp = reg.intercept_

        pendiente.append(m_temp)
        ordenada.append(b_temp)

    df_predict = pd.DataFrame(pendiente, ordenada)
    df_predict = df_predict.reset_index()

    df_predict.columns = ['intercept', 'slope']
    df_predict['slope'] = df_predict['slope']
    df_predict['rolling'] = df_predict['slope'].rolling(window=window).mean()
    return (df_predict)

def Clasifier_events(df):
    """
    This function assign a value of 1 if there is an event, or
    0 is not
    """
    df.loc[df['rolling'] >= 0.1, 'Event'] = 1.0
    df.loc[df['rolling'] < 0.1, 'Event'] = 0.0
    return df


def Join_DF_test(slope, dfE, dfN, dfU):
    """
    This function clean the main DataFrame from the first 70 points
    and then add the column Event.
    """
    dfE_2 = dfE.copy()
    dfN_2 = dfN.copy()
    dfU_2 = dfU.copy()

    slope_2 = slope.copy()

    dfE_2 = dfE_2.iloc[69:-35]
    dfN_2 = dfN_2.iloc[69:-35]
    dfU_2 = dfU_2.iloc[69:-35]
    slope_2 = slope_2.iloc[69:]

    dfE_2 = dfE_2.reset_index()
    dfN_2 = dfN_2.reset_index()
    dfU_2 = dfU_2.reset_index()
    slope_2 = slope_2.reset_index()

    dfE_2['n'] = pd.Series(dfN_2['x'])
    dfE_2['n_error'] = pd.Series(dfN_2['error'])
    dfE_2['u'] = pd.Series(dfU_2['x'])
    dfE_2['u_error'] = pd.Series(dfU_2['error'])
    dfE_2['Events'] = pd.Series(slope_2['Event'])

    final = dfE_2
    return (final)


def Input_Generator(df, input_data):
    """
    This function generate an input from all the station for a multi-layer perceptron NN
    """

    input_data = pd.concat([input_data, df[['x', 'n', 'u',  'Events']]])

    return (input_data)

# Get the data

for i in stations_list:
    base_url = "http://fits.geonet.org.nz/"
    endpoint = "observation"

    url = base_url + endpoint

    parameters = {"typeID": "e", "siteID": i}
    response_e = requests.get(url, params=parameters)
    parameters["typeID"] = "n"
    response_n = requests.get(url, params=parameters)
    parameters["typeID"] = "u"
    response_u = requests.get(url, params=parameters)

    globals()['stationsE_{0}'.format(i)] = GNSS_dataframe(response_e.content.decode("utf-8"))
    globals()['stationsE_{0}'.format(i)] = globals()['stationsE_{0}'.format(i)].set_index('date-time')
    globals()['stationsE_{0}'.format(i)].columns = ['x', 'error']

    globals()['stationsN_{0}'.format(i)] = GNSS_dataframe(response_n.content.decode("utf-8"))
    globals()['stationsN_{0}'.format(i)] = globals()['stationsN_{0}'.format(i)].set_index('date-time')
    globals()['stationsN_{0}'.format(i)].columns = ['x', 'error']

    globals()['stationsU_{0}'.format(i)] = GNSS_dataframe(response_u.content.decode("utf-8"))
    globals()['stationsU_{0}'.format(i)] = globals()['stationsU_{0}'.format(i)].set_index('date-time')
    globals()['stationsU_{0}'.format(i)].columns = ['x', 'error']


columns = [ 'x', 'n',  'u' ,'Events']
input_data = pd.DataFrame(columns=columns)

for i in stations_list:
#   Linear regression of data

    globals()['slope_{0}'.format(i)] = Regression_GPS(globals()['stationsE_{0}'.format(i)], 35)

#   Events classification

    globals()['slope_{0}'.format(i)] = Clasifier_events(globals()['slope_{0}'.format(i)])

#   Getting all the components and classification together

    globals()['final_{0}'.format(i)] = Join_DF_test(globals()['slope_{0}'.format(i)], globals()['stationsE_{0}'.format(i)]
                                                , globals()['stationsN_{0}'.format(i)]
                                                , globals()['stationsU_{0}'.format(i)])

#   Input data for a Multi-layer perceptron NN

    input_data = Input_Generator(globals()['final_{0}'.format(i)],input_data)


# Create a CSV file output
input_data = input_data.reset_index(drop=True)

# input_data.to_csv('/home/francovm/Projects/SSE/data/processed/GISB.csv', sep='\t', encoding='utf-8',index=False)

# Input data without errors
input_data.to_csv('/home/francovm/Projects/SSE/data/processed/input_data.csv', sep='\t', encoding='utf-8',index=False)