from .simulation import SimulationObject

class Crop(SimulationObject):
    def __init__(self, **kwargs):
        super(Crop, self).__init__(kwargs)
        pass

class CropConstant(Crop):
    def __init__(self, kcb, h, zr, fc):
        self._kcb = kcb
        self._height = h
        self._zr = zr
        self._fc = fc
    
    def kcb(self):
        return self._kcb
    
    def height(self):
        return self._height
    
    def root_depth(self):
        return self._zr
    
    def ground_covering(self):
        return self._fc
