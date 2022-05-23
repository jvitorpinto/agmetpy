import numpy as np

from .test_case_ndarray import TestCaseNDArray

from src.agmetpy import wb

class TestSoil(TestCaseNDArray):
    simulation = wb.Simulation(wb.CropConstant(kcb=0.8, h=0.5, zr=0.5, fc=0.8), wb.Soil(10 * [0.2], 0.3,0.2, 0.1, 0.1/86400, 0.1), weather=wb.Weather())

    def get_soil():
        soil = wb.Soil(10 * [0.2], 0.3,0.2, 0.1, 0.1/86400, 0.1)
        crop = wb.CropConstant(kcb=0.8, h=0.5, zr=0.5, fc=0.8)

    def test_properties(self):
        pass