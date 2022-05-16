import numpy as np

from .array_collection import ArrayCollection

def _format_arrays(**kwargs):
    for i in kwargs:
        kwargs[i] = kwargs[i] if len(np.shape(kwargs[i])) >= 2 else np.transpose(np.atleast_2d(kwargs[i]), (1, 0))
    return kwargs

class SimulationObject:
    
    def __init__(self, **kwargs):
        self._simulation: Simulation = kwargs.pop('simulation', None)
        self._var = ArrayCollection(**_format_arrays(**kwargs))
    
    def initialize(self):
        pass
    
    def update(self):
        pass

    def _get(self, varname):
        if varname in self._var:
            return self._var[varname]
        else:
            raise Exception(f'"{varname}" has not been assigned to object of type "{type(self)}"')
    
    def _set(self, varname, value):
        self._var[varname] = value
    
    def _copy(self, varname):
        return np.copy(self._get(varname))

    def _copyto(self, varname, value):
        np.copyto(self._var[varname], value)

    def _get_simulation(self) -> 'Simulation':
        if self._simulation is None:
            raise Exception('No simulation has been assigned')
        else:
            return self._simulation
        
    def _get_shape(self) -> tuple:
        return self._var.shape[1:]
    
    simulation = property(
        lambda self: self._get_simulation())
    
    shape = property(
        lambda self: self._get_shape())

class Simulation:

    def __init__(
            self,
            crop: 'crop.Crop',
            soil: 'soil.Soil',
            weather: 'weather.Weather'):
        self._crop = crop
        self._soil = soil
        self._weather = weather
        
        self._crop._simulation = self
        self._soil._simulation = self
        self._weather._simulation = self

        self._index: int = 0
        self._dt: int = 86400
    
    def __iter__(self):
        self._index = -1
        return self
    
    def __next__(self):
        self._index += 1
        if self.weather.finished():
            raise StopIteration()
        self.update()
        return self._index
    
    def update(self):
        self._weather.update()
        self._crop.update()
        self._soil.update()
    
    index = property(lambda self: self._index)

    weather = property(lambda self: self._weather)

    soil = property(lambda self: self._soil)

    crop = property(lambda self: self._crop)

    dt = property(lambda self: self._dt)

from . import crop
from . import soil
from . import weather