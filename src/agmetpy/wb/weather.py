
import numpy as np

from .simulation import SimulationObject

def maximum_kc(kcb, u2, rhmin, h):
    '''
    Calculates the maximum crop coefficient (kc) for a given period.

    Parameters
    ----------
    kcb : float
        Basal crop coeficient.
    u2 : float
        Wind speed [m/s].
    rhmin : float
        Relative humidity of air, between 0 and 1.
    h : float
        Crop height.

    Returns
    -------
    float
        Maximum crop coefficient (kc) for the period [adimensional].

    '''
    p1 = (0.04 * (u2 - 2)) - (0.4 * (rhmin - 0.45))
    p2 = (h / 3) ** 0.3
    return np.fmax(1.2 + p1 * p2, kcb + 0.05)

class Weather(SimulationObject):
    def __init__(
            self,
            repeat: bool = False,
            **kwargs):
        super(Weather, self).__init__(**kwargs)
        self._repeat = False
    

    def update(self):
        if self.index >= self.length:
            raise StopIteration()
    
    def _get_index(self):
        return self.simulation.index % self.length if self._repeat else self.simulation.index
    
    def __getitem__(self, key):
        if key in self._var:
            return self._var[key][self.index]
        else:
            raise Exception(f'"{key}" is not a valid name!')
    
    index = property(lambda self: self._get_index())

    length = property(lambda self: self._var.shape[0])
