import numpy as np

from .test_case_ndarray import TestCaseNDArray

from src.agmetpy import wb

class TestSoil(TestCaseNDArray):

    def test_fromto(self):
        soil = wb.Soil(
            10*[0.25],
            0.3,
            0.2,
            0.1,
            0.1/86400,
            0.125,
            ze=0.08,
            p=0.4,
            pe=0.28)

        soil.from_to(0.2, 0.45)

    def test_properties(self):
        soil = wb.Soil(
            10*[0.25],
            0.3,
            0.2,
            0.1,
            0.1/86400,
            0.125,
            ze=0.08,
            p=0.4,
            pe=0.28)
        
        self.assertNDArrayAlmostEqual(soil.theta, 0.25, 15)
        self.assertNDArrayAlmostEqual(soil._theta, 0.25, 15)
        self.assertNDArrayAlmostEqual(soil.theta_fc, 0.2, 15)
        self.assertNDArrayAlmostEqual(soil._theta_fc, 0.2, 15)
        self.assertNDArrayAlmostEqual(soil.theta_wp, 0.1, 15)
        self.assertNDArrayAlmostEqual(soil._theta_wp, 0.1, 15)
        self.assertNDArrayAlmostEqual(soil.theta_sat, 0.3, 15)
        self.assertNDArrayAlmostEqual(soil._theta_sat, 0.3, 15)
        self.assertNDArrayAlmostEqual(soil.ksat, 0.1/86400, 15)
        self.assertNDArrayAlmostEqual(soil._ksat, 0.1/86400, 15)
        self.assertNDArrayAlmostEqual(soil.dz, 0.125, 15)
        self.assertNDArrayAlmostEqual(soil._dz, 0.125, 15)
        self.assertNDArrayAlmostEqual(soil.theta_sat, 0.3, 15)
        self.assertNDArrayAlmostEqual(soil._theta_sat, 0.3, 15)
        self.assertNDArrayAlmostEqual(soil.p, 0.4, 15)
        self.assertNDArrayAlmostEqual(soil._p, 0.4, 15)

        self.assertNDArrayAlmostEqual(soil.ze, soil.from_to(0, 0.08), 15)
        self.assertNDArrayAlmostEqual(soil._ze, soil.from_to(0, 0.08), 15)

        self.assertEqual(soil.nlayers, 10)
        self.assertEqual(soil.depth, 1.25)
        self.assertEqual(soil.shape, (1,))
        self.assertEqual(soil.soil_shape, (10, 1))

        #dz = soil.from_to(0, soil.depth)
        #self.assertNDArrayAlmostEqual(soil.depletion_from_sat(dz), 0.00625, 15)
        #self.assertNDArrayAlmostEqual(soil.available_water(dz), 0.01875, 15)
        #self.assertNDArrayAlmostEqual(soil.depletion_from_fc(dz), 0, 15)
        #self.assertNDArrayAlmostEqual(soil.depletion_from_wp(dz), 0, 15)
        #self.assertNDArrayAlmostEqual(soil.total_available_water(dz), 0.0125, 15)
        #self.assertNDArrayAlmostEqual(soil.readily_available_water(dz), 0.005, 15)
        #self.assertNDArrayAlmostEqual(soil.total_evaporable_water(dz), 0.01875, 15)
        #self.assertNDArrayAlmostEqual(soil.readily_evaporable_water(dz), 0.01875, 15)
    
    def test_drainage(self):
        '''
        Test the drainage procedure and its results
        '''
        soil = wb.Soil(
            np.round([0.3 - i * 0.02 for i in range(10)], 2),
            0.3,
            0.2,
            0.1,
            0.1/86400,
            0.1)
        
        # A simulation is required because Soil.drainage_characteristic calls
        # self.simulation.dt. Crop, weather, and management instances are not
        # needed in the drainage procedure, so we initialize it without parameters.
        wb.Simulation(wb.Crop(), soil, wb.Weather(), wb.Management())
        
        self.assertNDArrayAlmostEqual(soil.drainage_characteristic(), 0.4340281443, 10)

        expected = np.array([[[0.30000000], [0.28000000], [0.26000000], [0.24000000], [0.22000000],
                              [0.20000000], [0.18000000], [0.16000000], [0.14000000], [0.12000000]],

                             [[0.25659719], [0.25659719], [0.25659719], [0.25659719], [0.25659719],
                              [0.21701407], [0.18000000], [0.16000000], [0.14000000], [0.12000000]],

                             [[0.23203241], [0.23203241], [0.23203241], [0.23203241], [0.23203241],
                              [0.23203241], [0.23203241], [0.21577310], [0.14000000], [0.12000000]],
                            
                             [[0.21812944], [0.21812944], [0.21812944], [0.21812944], [0.21812944],
                              [0.21812944], [0.21812944], [0.21812944], [0.21812944], [0.13683500]],
                              
                             [[0.21026076], [0.21026076], [0.21026076], [0.21026076], [0.21026076],
                              [0.21026076], [0.21026076], [0.21026076], [0.21026076], [0.20765320]],
                              
                             [[0.20580730], [0.20580730], [0.20580730], [0.20580730], [0.20580730],
                              [0.20580730], [0.20580730], [0.20580730], [0.20580730], [0.20580730]]])
        
        for i in range(6):
            self.assertNDArrayAlmostEqual(soil.theta, expected[i], 8)
            x = soil.drain(0)
        