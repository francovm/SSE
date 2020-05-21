from random import randint

from CGPM_ProjectManagement import CGPM_Class
from CGSIO_SimulationInputOutput import CGSIO_Class

import numpy as np
import matplotlib.pylab as plt

import pandas as pd

def getNextScenarioObject(pm, scenarioWildCardList, excludeName, layerName):
    #Replace this bit by a slip distribution generator
    sio=CGSIO_Class(pm)
    pm.gotoWorkingDirectory()
    scenarios = pm.listProjectsByWildCard(scenarioWildCardList, excludeByPart=excludeName)

    for scenario in scenarios:
        pm.switchToProject(scenario)
        sio.getComcotLayerDataAsImageNoInterpolation(layerName)
        yield (sio, scenario)

def timingRule01(d, zMax):
    return zMax/20*d

def genTimeSeries(displacementData):

    timeSeriesData = {}

    for scenario in displacementData:
        for station in displacementData[scenario]:
            timeSeriesData[station]=[0.0] #create a new entry with each station to hold time series as an empty list

    for scenario in displacementData:
        for station in displacementData[scenario]:
            numberOfQuietDays=randint(100, 1000)
            quietDays=np.linspace(start=0, stop=numberOfQuietDays, num=numberOfQuietDays)
            lastVal = timeSeriesData[station][-1]
            for d in quietDays:
                timeSeriesData[station].append(lastVal)

            zMax=displacementData[scenario][station]['zMax']
            days=np.linspace(start=0, stop=randint(15, 25), num=20)
            for d in days:
                z=timingRule01(d,zMax) #add the last displacement entry to get absolute displacements
                timeSeriesData[station].append(z+lastVal)

    return timeSeriesData

def runAll(pm, df):
    #this needs to loop and keep track of all Z values at the different locations
    displacementData = {}

    for index, (scenarioObject, scenarioName) in enumerate(getNextScenarioObject(pm, ['NAPIER_LEVEL*'],
                                                                                 excludeName='derived',
                                                                                 layerName = 'ini_surface.dat')):
        print('working on scenario: {}'.format(scenarioName))
        displacementData[scenarioName] = {}
        for station in df["Mark"].values:
            displacementData[scenarioName][station] = {}
            answer = df.loc[df['Mark'] == station][["Latitude", "Longitude"]]
            displacementData[scenarioName][station]['lat'] = answer['Latitude'].values[0]
            displacementData[scenarioName][station]['lon'] = answer['Longitude'].values[0]
            displacementData[scenarioName][station]['zMax'] = scenarioObject.getLayerDataValue(answer['Longitude'].values[0],answer['Latitude'].values[0])

    return genTimeSeries(displacementData)

if __name__ == '__main__':
    #replace this bit by site locations
    df=pd.read_csv('/home/cmu/workspace/SSE/data/external/stations_locations.csv',sep='\t')

    configFile='./config/slowSlip.JSON'
    pm=CGPM_Class(configFile=configFile,setup=False,doCluster=False,update=False)

    timeSeriesData = runAll(pm, df)

    for station in timeSeriesData:
        plt.plot(timeSeriesData[station])
    plt.show()