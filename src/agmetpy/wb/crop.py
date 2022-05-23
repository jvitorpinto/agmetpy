from .simulation import SimulationObject
import numpy as np

import numpy as np

def _safe_divide(a, b, on_zero=np.inf):
    zerodiv = b == 0
    div = np.where(zerodiv, 1, b)
    return np.where(zerodiv, on_zero, a / div)

class RootDistribution:

    def __init__(self, zr, zrmax, k):
        self._zr = zr
        self._zrmax = zrmax
        self.k = k

    def __call__(self, z1, z2):
        zr = self.zr
        zrmax = self.zrmax
        
        sqinv = _safe_divide(zrmax, zr, 1)
        zerodiv = zr == 0

        k1 = self.k
        k2 = (sqinv - k1) * sqinv
        
        zrel = lambda z: _safe_divide(np.minimum(z, zr), zrmax, 1)
        poly = lambda x: (k1 + k2 * x) * x

        return np.where(zerodiv, 0, poly(zrel(z2)) - poly(zrel(z1)))

    def _get_zr(self):
        return self._zr() if callable(self._zr) else self._zr
    
    def _get_zrmax(self):
        return self._zrmax() if callable(self._zrmax) else  self._zrmax
    
    zr = property(lambda self: self._get_zr())

    zrmax = property(lambda self: self._get_zrmax())

class Crop(SimulationObject):

    root_dist = RootDistribution(
        zr=lambda self: self._get_zr(),
        zrmax=lambda self: self._get_zrmax(),
        k=1.8)
    
    def __init__(self, **kwargs):
        super(Crop, self).__init__(**kwargs)

    def _get_kcb(self):
        return self._get('kcb')
    
    def _get_height(self):
        return self._get('h')

    def _get_zr(self):
        return self._get('zr')
    
    def _get_zrmax(self):
        return self._get('zr')
    
    def _get_ground_covering(self):
        return self._get('fc')
    
    kcb = property(
        lambda self: self._get_kcb())

    height = property(
        lambda self: self._get_height())

    zr = property(
        lambda self: self._get_zr())

    zrmax = property(
        lambda self: self._get_zrmax())

    ground_covering = property(
        lambda self: self._get_ground_covering())

class CropConstant(Crop):
    def __init__(self, kcb, h, zr, fc, repeat: bool = True):
        super(CropConstant, self).__init__(kcb=kcb, h=h, zr=zr, fc=fc)
        self.repeat = repeat
    
    def _get_index(self):
        i = super(CropConstant, self)._get_index()
        return i % self._var.shape[0] if self.repeat else i
    
    def _get(self, varname):
        return super(CropConstant, self)._get(varname)[self.index]
    
    def _copy(self, varname):
        return super(CropConstant, self)._copy(varname)[self.index]
