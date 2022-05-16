import unittest
import numpy as np

from src.agmetpy import wb

class TestCaseNDArray(unittest.TestCase):
    def assertNDArrayEqual(self, a, b):
        self.assertTrue(np.all(a == b))
    
    def assertNDArrayAlmostEqual(self, a, b, decimals):
        self.assertTrue(np.all(np.round(a, decimals) == np.round(b, decimals)))
