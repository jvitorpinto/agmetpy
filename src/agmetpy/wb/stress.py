import numpy as np

class Stress:
    def __call__(self, x=None):
        return np.ones_like(x)

class StressResponse(Stress):
    def __init__(self, xmin, xmax, ymin = 0, x = None, reverse = False) -> None:
        self.x = x
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.reverse = reverse

    def get_x(self):
        return self._x() if callable(self._x) else x
    
    def set_x(self, value):
        self._x = value
    
    def get_xmin(self):
        return self._xmin() if callable(self._xmin) else self._xmin
    
    def set_xmin(self, value):
        self._xmin = value
    
    def get_xmax(self):
        return self._xmax() if callable(self._xmax) else self._xmax
    
    def set_xmax(self, value):
        self._xmax = value
    
    def get_ymin(self):
        return self._ymin() if callable(self._ymin) else self._ymin
    
    def set_ymin(self, value):
        self._ymin = value
    
    x = property(lambda self: self.get_x(), lambda self, value: self.set_x(value))

    xmin = property(lambda self: self.get_xmin(), lambda self, value: self.set_xmin(value))

    xmax = property(lambda self: self.get_xmax(), lambda self, value: self.set_xmax(value))

    ymin = property(lambda self: self.get_ymin(), lambda self, value: self.set_ymin(value))

class StressLinear(StressResponse):
    def __call__(self, x=None):
        dx = self.xmax - self.xmin
        k = self.x - self.xmin if x is None else x - self.xmin
        k = np.clip(k/dx, 0, 1)
        
        if self.reverse:
            k = 1-k
        
        return (self.ymin) + (1-self.ymin) * k

class StressCombined(Stress):
    def __init__(self, *args: Stress) -> None:
        self._stresses = args
        super(StressCombined, self).__init__()
