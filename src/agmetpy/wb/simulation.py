from calendar import c
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
    
    def _get_index(self):
        return self.simulation.index
    
    simulation = property(
        lambda self: self._get_simulation())
    
    shape = property(
        lambda self: self._get_shape())
    
    index = property(
        lambda self: self._get_index())

class Simulation:

    def __init__(
            self,
            crop: 'crop.Crop',
            soil: 'soil.Soil',
            weather: 'weather.Weather',
            management: 'Management'):

        self._crop = self.assign(crop)
        self._soil = self.assign(soil)
        self._weather = self.assign(weather)
        self._management = self.assign(management)
        
        self._index: int = 0
        self._dt: int = 86400
    
    def assign(self, obj: SimulationObject):
        obj._simulation = self
        return obj
    
    def __iter__(self):
        self._index = -1
        return self
    
    def __next__(self):
        self._index += 1
        self.update()
        return self._index
    
    def update(self):
        self._weather.update()
        self._management.update()
        self._crop.update()
        self._soil.update()
    
    index = property(lambda self: self._index)

    weather = property(lambda self: self._weather)

    soil = property(lambda self: self._soil)

    crop = property(lambda self: self._crop)

    management = property(lambda self: self._management)

    dt = property(lambda self: self._dt)

class Management(SimulationObject):

    '''
    Wetted soil fraction
    '''
    fw = property(lambda self: self.get_fw())

    '''
    Exposed and wetted soil fraction
    '''
    few = property(lambda self: self.get_few())

    irrigation = property(lambda self: self.get_irrigation())

class ManagementConstant(Management):
    def __init__(self, fw_ini, repeat=True, **kwargs):
        self.repeat = repeat
        self._fw = fw_ini
        super(ManagementConstant, self).__init__(**kwargs)
    
    def get_irri_fw(self):
        return np.ones(self.shape)
    
    def get_fw(self):
        rain = self.simulation.weather['rainfall']
        irri = self.irrigation
        irri_fw = self.get_irri_fw()
        total = rain + irri
        zerodiv = total == 0
        total = np.where(zerodiv, 1, total)
        fw = np.where(zerodiv, self._fw, (rain + irri_fw * irri) / total)
        return fw
    
    def _get_index(self):
        i = super(ManagementConstant, self)._get_index()
        return i % self._var.shape[0] if self.repeat else i
    
    def get_few(self):
        fc = self.simulation.crop.ground_covering
        fw = self.get_fw()
        return np.minimum(1-fc, fw)
    
    def get_irrigation(self):
        return self._get('irrigation')[self.index]

from . import crop
from . import soil
from . import weather