from CGPM_ProjectManagement import CGPM_Class
from CGSIO_SimulationInputOutput import CGSIO_Class
import numpy as np
import matplotlib.pylab as plt

configFile = './config/slowSlip.JSON'
pm = CGPM_Class(configFile=configFile, setup=False, doCluster=False, update=False)

pm.switchToProject('testCase')


#Replace this bit by a slip distribution generator
sio = CGSIO_Class(pm)
image = sio.getComcotLayerDataAsImageNoInterpolation('ini_surface_layer02.dat.gz')


#replace this bit by site locations
x = 175.0
y = -41.0

#this needs to loop and keep track of all Z values at the different locations
zMax = sio.getLayerDataValue(x,y)

#this needs to be replcaed witha propoer time series generator based on the list of z-values from above
days = np.linspace(start=0, stop=19, num=20)
zValues = []
for d in days:

    z = zMax/20 * d
    zValues.append(z)

#this has to be replaced by a data storage thing
plt.plot(days, zValues, '*')
plt.show()


