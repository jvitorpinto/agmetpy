# %%
import xarray as xr
import numpy as np
from xarray import Dataset, DataArray

def as_dist(array: DataArray, dim: str = 'z'):
    total = array.sum(dim)
    total = total.where(total != 0, 1)
    return array / total

# def _safe_divide(a, b, on_zero = np.inf):
#     zerodiv = b == 0
#     div = np.where(zerodiv, 1, b)
#     return np.where(zerodiv, on_zero, a / div)

def _safe_divide(a: DataArray, b: DataArray, on_zero = np.inf):
    zerodiv = b == 0
    div = xr.where(zerodiv, 1, b)
    return xr.where(zerodiv, on_zero, a / div)

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

class Soil:

    def __init__(self, array: Dataset) -> None:
        self.array = array
        pass

    def get_theta_sat(self) -> DataArray:
        return self.array.theta_sat
    
    def get_theta_fc(self) -> DataArray:
        return self.array.theta_fc
    
    def get_theta_wp(self) -> DataArray:
        return self.array.theta_wp
    
    def get_theta(self) -> DataArray:
        return self.array.theta
    
    def get_dz(self) -> DataArray:
        return self.array.z - self.array.z.shift(z=1, fill_value=0)
    
    def get_depth(self) -> DataArray:
        return self.array.z.max('z')
    
    def get_ze(self) -> DataArray:
        return self.array.ze
    
    def get_upper_layer_boundary(self):
        return self.array.z.shift(z=1, fill_value=0)
    
    def get_bottom_layer_boundary(self):
        return self.array.z
    
    def total_available_water(self, zr) -> DataArray:
        '''
        Calculates the total available water (taw).

        TAW is calculated as:

            taw = zr * (theta_sat - theta_wp)

        Where:
            - `taw`: total available water in m.
            - `zr`: root depth (m).
            - `theta_sat`: soil moisture at saturation.
            - `theta_wp`: soil moisture at wilting point.
        
        Parameters
        ----------
        zr: xarray.DataArray
            An array containing the root zone depth in m.
        
        Returns
        -------
        xarray.DataArray
            An array containing the taw for each cell of the soil
            reached by the root zone (m).
        '''
        return self.from_to(0, zr) * (self.theta_sat - self.theta_wp)
    
    def total_evaporable_water(self) -> DataArray:
        return self.from_to(0, self.ze) * (self.theta_sat - 0.5*self.theta_wp)
    
    def readily_evaporable_water(self) -> DataArray:
        return self.from_to(0, self.ze) * ()
    
    def depletion(self) -> DataArray:
        '''
        Calculates the depletion of each cell.

        Depletion is:

        dp = dz * (theta_sat - theta)

        All values are guaranteed to be >= 0 as much as
        theta_sat >= theta.

        Returns
        -------
        xarray.DataArray
            A DataArray instance with the same shape as soil.shape
            with the water depletion of each soil cell, in m.
        '''
        return self.dz * (self.theta_sat - self.theta)
    
    def infiltrate(self, water_depth: DataArray) -> Dataset:
        '''
        Calculates the new soil moisture after infiltrating `water_depth` and
        returns the variation of soil moisture (m³/m³) as well as the deep
        percolation (m).
        '''
        depletion = self.depletion()
        deep_percolation = water_depth
        dtheta = []
        for _, dp in depletion.groupby('z', squeeze=False):
            var = np.minimum(dp, deep_percolation)
            # deep_percolation has no z coordinate,
            # but var has, so we select the first dimension
            # of z to make sure it will not have a z dimension
            deep_percolation = (deep_percolation - var).isel(z=0)
            dtheta.append(var)
        dtheta = xr.concat(dtheta, 'z') / self.dz
        return Dataset(data_vars={
            'var_theta': dtheta,
            'deep_percolation': deep_percolation
        })

    def from_to(self, zmin, zmax):
        if np.any(zmin > zmax):
            raise Exception('zmin cannot be greater than zmax')
        if np.any(zmax > self.depth):
            raise Exception('zmax cannot be greater than Soil.depth')
        zini = self.get_upper_layer_boundary()
        dz = self.dz
        return (-zini + zmax).clip(0, dz) - (zmin - zini).clip(0, dz)
    
    theta_sat: DataArray = property(lambda self: self.get_theta_sat())
    '''
    Gets the soil moisture at saturation [m³/m³].
    '''

    theta_fc: DataArray = property(lambda self: self.get_theta_fc())
    '''
    Gets the soil moisture at field capacity [m³/m³].
    '''

    theta_wp: DataArray = property(lambda self: self.get_theta_wp())
    '''
    Gets the soil moisture at wilting point [m³/m³].
    '''

    theta: DataArray = property(lambda self: self.get_theta())
    '''
    Gets the current soil moisture [m³/m³].
    '''

    dz: DataArray = property(lambda self: self.get_dz())
    '''
    Gets the thicknes of each soil layer [m].
    '''

    ze: DataArray = property(lambda self: self.get_ze())
    '''
    Gets the thickness of the evaporative layer [m].
    '''

    depth: DataArray = property(lambda self: self.get_depth())
    '''
    Gets the depth of the simulated soil.
    '''

class Crop:

    def __init__(self, array: Dataset, root_shape: np.float64 = 1.8) -> None:
        '''
        
        Parameters
        ----------
        array: Dataset
            A xarray.Dataset containing at least the following
            variables:
            - kc: the crop coefficient (adimensional)
            - root_depth: the effective root depth (m)
            - height: the crop height above ground (m)
        '''

        self.array: Dataset = array
        ''' The array containing the crop's data '''

        self.root_shape: np.float64 = root_shape
        '''
        The shape of root distribution function.
        Must be between 1 (continuous distribution) and 2
        (more concentrated near the surface). The default is
        1.8.
        '''
        pass

    def root_dist(self, z_ini, z_end):
        '''
        Calculates the root distribution for this crop.

        Parameters
        ----------
        z_ini: DataArray
            Initial root depth

        z_end: DataArray
            Final root
        
        '''
        if np.any(z_ini > z_end):
            raise Exception('z_ini must always be less than or equal to z_end')

        zr = self.root_depth
        zrmax = self.max_root_depth

        sqinv = _safe_divide(zrmax, zr, 0)
        print(type(sqinv))
        zerodiv = zr == 0

        k1 = self.root_shape
        k2 = (sqinv - k1) * sqinv
        
        zrel = lambda z: _safe_divide(np.minimum(z, zr), zrmax, 1)
        poly = lambda x: (k1 + k2 * x) * x

        return xr.where(zerodiv, 0, poly(zrel(z_end)) - poly(zrel(z_ini)))

    def get_root_depth(self) -> DataArray:
        return self.array.root_depth
    
    def get_max_root_depth(self) -> DataArray:
        return self.array.root_depth
    
    def get_kc(self) -> DataArray:
        return self.array.kc
    
    def get_height(self) -> DataArray:
        return self.array.height
    
    root_depth: DataArray = property(lambda self: self.get_root_depth())
    '''
    Gets the root depth of the crop.
    '''

    max_root_depth: DataArray = property(lambda self: self.get_max_root_depth())
    '''
    Gets the maximum root depth.
    '''

    kc: DataArray = property(lambda self: self.get_kc())
    '''
    Gets the crop coefficient (kc).
    '''

    height: DataArray = property(lambda self: self.get_height())
    '''
    Gets the crop's height.
    '''

class Weather:

    def __init__(self) -> None:
        pass

class Model:

    def __init__(self) -> None:
        pass

soil_array = Dataset(
    data_vars={
        'theta_sat': DataArray(np.full((10, 2, 2), 0.25), dims=('z', 'lat', 'lon')),
        'theta_fc': DataArray(np.full((10, 2, 2), 0.22), dims=('z', 'lat', 'lon')),
        'theta_wp': DataArray(np.full((10, 2, 2), 0.08), dims=('z', 'lat', 'lon')),
        'theta': DataArray(np.full((10, 2, 2), 0.2), dims=('z', 'lat', 'lon')),
        'ze': DataArray(np.full((2, 2), 0.1), dims=('lat', 'lon')),
    },
    coords={
        'z': np.arange(0, 1, 0.1) + 0.1,
        'lat': [-2, -3],
        'lon': [-48, -49]
    }
)

soil = Soil(soil_array)

rainfall = DataArray(
    [[0.013, 0.025], [0.025, 0.06]],
    dims=('lat', 'lon'),
    coords={
        'lat': DataArray([-2, -3], dims=('lat',)),
        'lon': DataArray([-48, -49], dims=('lon')),
    }
)

zr = DataArray(
    data=[
        [0.15, 0.25],
        [0.25, 0.4],
    ],
    dims=('lat', 'lon'),
    coords={
        'lat': DataArray([-2, -3], dims=('lat',)),
        'lon': DataArray([-48, -49], dims=('lon')),
    }
)

crop = Crop(Dataset({'root_depth': zr}))
dist = crop.root_dist(
    soil.get_upper_layer_boundary(),
    soil.get_bottom_layer_boundary()
)
dist = dist.transpose('z', 'lat', 'lon')
dist
# %%
