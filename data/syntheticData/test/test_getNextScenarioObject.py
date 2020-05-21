from unittest import TestCase
from CGPM_ProjectManagement import CGPM_Class
from data.syntheticData import slipGenerator
from data.syntheticData.slipGenerator import getNextScenarioObject


class TestGetNextScenarioObject(TestCase):

	def setUp(self):
		configFile='../config/slowSlip.JSON'
		self.pm=CGPM_Class(configFile=configFile,setup=False,doCluster=False,update=False)

	def test_getNextScenarioObject(self):
		answer=getNextScenarioObject(self.pm,['NAPIER_LEVEL*'],
									 excludeName='derived',
									 layerName='ini_surface.dat')

		self.assertIsNotNone(answer)

		for sio in answer:
			self.assertIsNotNone(sio)
			break
