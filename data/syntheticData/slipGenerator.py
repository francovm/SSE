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

def genTimeSeries(displacementData, scenarioName):
    #this needs to be replcaed witha propoer time series generator based on the list of z-values from above
    days=np.linspace(start=0,stop=19,num=20)
    for key in displacementData[scenarioName]:
        zMax=displacementData[scenarioName][key]['zMax']
        zValues=[]
        for d in days:
            z=timingRule01(d,zMax)
            zValues.append(z)
        displacementData[scenarioName][key]['timeSeries']=zValues

    return displacementData

def runAll(pm, df):
    #this needs to loop and keep track of all Z values at the different locations
    displacementData = {}

    for index, (scenarioObject, scenarioName) in enumerate(getNextScenarioObject(pm, ['NAPIER_LEVEL*'],
                                                                                 excludeName='derived',
                                                                                 layerName = 'ini_surface.dat')):
        print('working on scenario: {}'.format(scenarioName))
        displacementData[scenarioName] = {}
        for name in df["Mark"].values:
            displacementData[scenarioName][name] = {}
            answer = df.loc[df['Mark'] == name][["Latitude", "Longitude"]]
            displacementData[scenarioName][name]['lat'] = answer['Latitude'].values[0]
            displacementData[scenarioName][name]['lon'] = answer['Longitude'].values[0]
            displacementData[scenarioName][name]['zMax'] = scenarioObject.getLayerDataValue(answer['Longitude'].values[0],answer['Latitude'].values[0])

        displacementData = genTimeSeries(displacementData,scenarioName)


if __name__ == '__main__':
    #replace this bit by site locations
    df=pd.read_csv('/home/cmu/workspace/SSE/data/external/stations_locations.csv',sep='\t')

    configFile='./config/slowSlip.JSON'
    pm=CGPM_Class(configFile=configFile,setup=False,doCluster=False,update=False)

    runAll(pm, df)

    pass