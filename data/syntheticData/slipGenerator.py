from CGPM_ProjectManagement import CGPM_Class
from CGSIO_SimulationInputOutput import CGSIO_Class
import numpy as np
import matplotlib.pylab as plt

import pandas as pd

configFile = './config/slowSlip.JSON'
pm = CGPM_Class(configFile=configFile, setup=False, doCluster=False, update=False)

pm.switchToProject('testCase')
#Replace this bit by a slip distribution generator
sio = CGSIO_Class(pm)
sio.getComcotLayerDataAsImageNoInterpolation('ini_surface_layer02.dat.gz')

#replace this bit by site locations
df = pd.read_csv('/home/cmu/workspace/SSE/data/external/stations_locations.csv', sep='\t')

#this needs to loop and keep track of all Z values at the different locations
displacementData = {}

for name in df["Mark"].values:
    displacementData[name] = {}
    answer = df.loc[df['Mark'] == name][["Latitude", "Longitude"]]
    displacementData[name]['lat'] = answer['Latitude'].values[0]
    displacementData[name]['lon'] = answer['Longitude'].values[0]
    displacementData[name]['zMax'] = sio.getLayerDataValue(answer['Longitude'].values[0],answer['Latitude'].values[0])

#this needs to be replcaed witha propoer time series generator based on the list of z-values from above
days = np.linspace(start=0, stop=19, num=20)


def timingRule01(d):
    return zMax/20*d


for key in displacementData:
    zMax = displacementData[key]['zMax']
    zValues=[]
    for d in days:
        z = timingRule01(d)
        zValues.append(z)
    displacementData[key]['timeSeries'] = zValues

# #this has to be replaced by a data storage thing
for key in displacementData:
    zValues = displacementData[key]['timeSeries']
    plt.plot(days, zValues)
    plt.hold(True)

plt.show()


