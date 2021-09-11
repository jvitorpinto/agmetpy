# -*- coding: utf-8 -*-

'''
Created on Sep 8 2021

@author João Vitor de Nóvoa Pinto (jvitorpinto@gmail.com)
'''

import numpy as np

class Stress:
    def coefficient(self, x):
        '''
        Calculates the reduction coefficient as a function
        of the stressing factor x.

        Parameters
        ----------
        x: array_like, float
            Stressing factor.
        '''
        pass

class StressResponse(Stress):
    '''
    Represents a generic response to a stressing factor.
    '''
    def __init__(self, xmin, xmax, ymin=0, inverse=False) -> None:
        '''
        Initializes a StressResponse object.

        Parameters
        ----------
        xmin: array_like, float
            The value of the stressing factor at which the stress response
            begins
        xmax: array_like, float
            The value of the stressing factor at which the stress response
            reaches its maximum.
        ymin: array_like, float
            The smallest stress reduction coefficient.
        inverse: array_like, bool

        '''
        super().__init__()
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._inverse = inverse
    
    def get_xmin(self):
        return self._xmin
    
    def set_xmin(self, value):
        self._xmin = value
    
    xmin = property(get_xmin, set_xmin)

class StressLinear(StressResponse):
    '''
    Represents a linear response to a stressing factor.
    '''
    def coefficient(self, x):
        dx = self._xmax - self._xmin
        # if (xmax - xmin) / dx will result in a zero-division error
        # then dx is set to 1, the result is calculated and then the
        # result is set to 1 where dx was 0.
        zerodiv = dx == 0
        dx = np.where(zerodiv, 1, dx)
        y = np.clip((x - self._xmin) / dx, 0, 1)
        y = np.where(zerodiv, np.where(x > self._xmax, 1, 0), y)
        y = (1-self._ymin) * y
        if self._inverse:
            return 1 - y
        else:
            return self._ymin + y

class StressCombined(Stress):
    def __init__(self, stresses):
        super().__init__()
        self._stresses = stresses

class StressCombinedMinimum(StressCombined):
    def coefficient(self, x):
        y = 1
        for stress in self._stresses:
            y = np.minimum(y, stress.coefficient(x))
        return y

class StressCombinedProduct(StressCombined):
    def coefficient(self, x):
        y = 1
        for stress in self._stresses:
            y = y * stress.coefficient(x)
        return y
