import numpy as np
import xarray as xr
from xarray import DataArray, Dataset
from typing import Self

class Weather:

    def __init__(
            self, dataset: Dataset,
            dim_name: str, sequential: bool = False,
            max_iter: int = 10
        ):
        self.groups = list(dataset.groupby(dim_name))
        self.dim_name = dim_name
        self.pr_name = 'pr'
        self.etp_name = 'etp'
        self._index = None
        self.sequential = sequential
        self.max_iter = max_iter

    def has_finished(self) -> bool:
        if self.sequential:
            return self.index >= self.size
        else:
            return self.index >= self.max_iter * self.size
    
    def read(self) -> Dataset:
        return self.groups[self.period][1]

    def __iter__(self) -> Self:
        self._index = -1
        return self
    
    def __next__(self) -> tuple[int, Dataset]:
        self.update()
        if self.has_finished():
            raise StopIteration()
        else:
            return (self.index, self.read())
    
    def update(self):
        self._index += 1
    
    def get_index(self) -> int:
        return self._index
    
    def get_period(self) -> int:
        if self.sequential:
            return self._index
        else:
            return self._index % self.get_size()
    
    def get_size(self) -> int:
        return len(self.groups)

    index = property(lambda self: self.get_index())

    period = property(lambda self: self.get_period())

    size = property(lambda self: self.get_size())

class Simulation():

    def __init__(self, weather: Weather) -> None:
        self.weather = weather
        pass

    def run(self):
        for _, data in self.weather:
            pass


from .. import toolbox

def thornthwaite_evapotranspiration(temp, time_dim = 'month'):
    '''
    Calculates the evapotranspiration
    '''
    temp = np.maximum(temp, 0)
    heat_index = ((temp / 5) ** 1.514).sum(time_dim)
    k3, k2, k1, k0 = 6.75e-7, -7.71e-5, 1.79e-2, 0.49
    a = k0 + (k1 + (k2 + k3*heat_index) * heat_index) * heat_index
    # if T <= 0°C for all months the heat index will be 0,
    # so a zerodiv error will happen here.
    cond1 = 16 * (10 * temp / heat_index) ** a
    cond2 = (32.24-0.43*temp)*temp - 415.85
    return xr.where(temp < 26.5, cond1, cond2)
