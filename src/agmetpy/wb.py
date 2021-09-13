import numpy as np
from . import stress

class Event:
    def __init__(self, parent):
        self._parent = parent
        self._subscriptions = []
    
    def subscribe(self, event_handler):
        self._subscriptions.append(event_handler)
    
    def unsubscribe(self, event_handler):
        self._subscriptions.remove(event_handler)
    
    def dispatch(self, event_args):
        for event_handler in self._subscriptions:
            event_handler(self._parent, event_args)

class SimulationObject:
    '''
    Represents an object that is initialized and then updated
    at each step of a simulation.
    '''
    def __init__(self):
        self._simulation = None
    
    def initialize(self):
        pass
    
    def update(self):
        pass
    
class Crop(SimulationObject):
    def __init__(self):
        pass
    
    def kcb(self):
        pass
    
    def height(self):
        pass
    
    def root_depth(self):
        pass
    
    def ground_covering(self):
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

class CropNone(CropConstant):
    def __init__(self):
        super(CropNone, self).__init__(kcb = 0, h = 0, zr = 0, fc = 0)

class Soil(SimulationObject):
    def __init__(self, theta, theta_sat, theta_fc, theta_wp, dz=0.1, ze=0.1):
        '''
        Represents a soil.
        
        Parameters
        ----------
        theta : np.ndarrray(dtype = "float")
            Initial soil moisture [m³/m³].

        theta_sat : float, array_like
            Soil moisture at saturation [m³/m³].

        theta_fc : float, array_like
            Soil moisture at field capacity [m³/m³].

        theta_wp : float, array_like
            Soil moisture at wilting point [m³/m³].

        dz : float, array_like, optional
            Thickness of soil layers [m]. The default is 0.1.

        ze : float, array_like, optional
            Thickness of evaporation layer [m]. The default is 0.1.

        p : float, array_like, optional
            DESCRIPTION. The default is 0.5.
        '''
        # soil moisture [m³/m³]
        self._theta = theta
        # soil moisture at saturation [m³/m³]
        self._theta_sat = theta_sat
        # soil moisture at field capacity [m³/m³]
        self._theta_fc = theta_fc
        # soil moisture at wilting point [m³/m³]
        self._theta_wp = theta_wp
        # layer height [m]
        self._dz = dz
        # height of evaporative layer [m]
        self._ze = ze
        # depth of soil [m]
        self._depth = theta.shape[0] * dz
    
    def depletion_from(self, theta_ref, z=None, z0=0, total=False):
        de = np.fmax(theta_ref - self._theta, 0.) * self.from_to(z, z0)
        if total:
            de = de.sum(0)
        return de
    
    def depletion_from_sat(self, z=None, z0=0, total=False):
        return self.depletion_from(self._theta_sat, z, z0, total)
    
    def depletion_from_fc(self, z=None, z0=0, total=False):
        return self.depletion_from(self._theta_fc, z, z0, total)
    
    def depletion_from_wp(self, z=None, z0=0, total=False):
        '''
        Calculates the amount of water depleted from wilting point
        between z0 and z.

        Parameters
        ----------
        z0: float, array_like
            The uppermost point of the soil profile.

        z: float, array_like
            The lowest point.
        
        total: boolean
            If total=True depletion of all soil layers is summed, otherwise
            the depletion of each layer will be returned.
        '''
        return self.depletion_from(self._theta_wp, z, z0, total)
    
    def depletion_of_evap_layer(self, total=False):
        return self.depletion_from_fc(self._ze, total=total)
    
    def depletion_of_root_zone(self, zr, total=False):
        return self.depletion_from_fc(zr, total=total)
    
    def available_water(self, theta, theta_ref, z=None, z0=0, total=False):
        aw = np.fmax((theta - theta_ref) * self.from_to(z, z0), 0.)
        if total:
            aw = aw.sum(0)
        return aw
    
    def total_available_water(self):
        zr = self.simulation.root_depth()
        return self.available_water(self._theta_fc, self._theta_wp, zr)
    
    def evaporable_water(self, total):
        return self.available_water(self._theta, 0.5 * self._theta_wp, z=self._ze, total=total)
    
    def total_evaporable_water(self, total):
        return self.available_water(self._theta_fc, 0.5 * self._theta_wp, self._ze, total=total)
    
    def from_to(self, z, z0 = 0):
        return z - z0

class SoilLayered(Soil):
    def __init__(self, theta, theta_sat, theta_fc, theta_wp, dz = 0.1, ze = 0.1, p = 0.5):
        super(SoilLayered, self).__init__(theta, theta_sat, theta_fc, theta_wp, dz, ze)
        self._p = p
        self._pe = 0.3
    
    def initialize(self):
        # self._ks holds a reference to self._taw and self._raw, so both self._taw
        # and self._raw must be updated in place. The same occurs for self._kr.
        zr = self.simulation.root_depth()
        
        self._taw = np.atleast_1d(self.available_water(self._theta_fc, self._theta_wp, zr, total=True))
        self._raw = self._p * self._taw
        self._ks = stress.StressLinear(self._raw, self._taw)
        
        self._tew = np.atleast_1d(self.available_water(self._theta_fc, 0.5 * self._theta_wp, self._ze, total=True))
        self._rew = self._pe * self._tew
        self._kr = stress.StressLinear(self._rew, self._tew)
        # end initialize(self)
    
    def _update_taw(self, zr):
        taw = np.atleast_1d(self.available_water(self._theta_fc, self._theta_wp, zr, total=True))
        raw = self._p * self._taw
        
        np.copyto(self._taw, taw)
        np.copyto(self._raw, raw)
    
    def update(self):
        # updates the root zone
        zr = self.simulation.root_depth()
        
        # chuva + irrigação
        water_depth = (self.simulation.rainfall() + self.simulation.irrigation()) / 1000.
        self._increase_theta_from_top(water_depth)
        
        # retrieves reference evapotranspiration from simulation data
        # Crop evapotranpiration:
        #    et = ks * kc * et0
        # Evaporation:
        #    ev = kr * ke * et0
        et0 = self.simulation.ref_et() / 1000.
        
        de = self.depletion_of_evap_layer(total=True)
        ke = np.fmin(self.simulation.ke(), self.simulation.f_ew() * self.simulation.maximum_kc())
        self.simulation._set_actual_evaporation(self._kr.coefficient(de) * ke * et0)
        self.extract_water(ke * et0,
                           self._theta_fc, 0.5 * self._theta_wp, self._ze, 0.,
                           self._kr.coefficient(de))
        
        kb = self.simulation.kcb()
        zr = self.simulation.root_depth()
        dr = self.depletion_from_fc(zr, total=True)
        self._update_taw(zr)
        self.simulation._set_actual_transpiration(self._ks.coefficient(dr) * kb * et0)
        self.extract_water(kb * et0,
                           self._theta_fc, self._theta_wp, zr, 0.,
                           self._ks.coefficient(dr))
    
    def extract_water(self, depth, theta_max, theta_min, z, z0, kr):
        '''
        Extracts water between a soil region defined by z0 and z, and limited
        to the moisture interval defined by [theta_min, theta_max].

        Parameters
        ----------
        depth : array-like float
            Depth of water to be extracted.
        theta_max : float
            DESCRIPTION.
        theta_min : TYPE
            DESCRIPTION.
        z : float
            DESCRIPTION.
        z0 : float
            DESCRIPTION.
        kr : float
            A reduction coefficient between 0 and 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # water available between theta_max and theta_min [m]
        taw = self.available_water(theta_max, theta_min, z, z0)
        # water available between theta and theta_min [m]
        aw = np.fmin(self.available_water(self._theta, theta_min, z, z0), taw)
        weights = SoilLayered.as_weights(aw)
        # the total extracted water is limited by the water available between z0 and z1
        ext_total = np.fmin(kr * depth, aw.sum(0))
        # then the extraction of water is distributed through layers by a weighting
        # factor calculated from the available water in each layer.
        ext = weights * ext_total
        self._theta -= ext / self._dz
        return depth - ext_total
    
    def _increase_theta_from_top(self, depth):
        # mudar isso depois
        dp = self.depletion_from_fc()
        dp -= np.clip(depth - dp.cumsum(0), 0., dp)
        self._theta = np.clip(self._theta_fc - dp /
                              self._dz, 0, self._theta_fc)
    
    def from_to(self, z=None, z0=0):
        '''
        Parameters
        ----------
        z : TYPE, optional
            DESCRIPTION. The default is None.

        z0 : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if z is None:
            z = self._depth
        dz, one = self._dz, np.ones(self._theta.shape)
        depth = (dz * one).cumsum(0) - dz
        dz1 = np.clip((z0 * one) - depth, 0, dz)
        dz2 = np.clip((z * one) - depth, 0, dz)
        return dz2 - dz1
    
    def weights(self, z=None, z0=0):
        dz = self.from_to(z, z0)
        return Soil.as_weights(dz)
    
    def as_weights(x):
        total = x.sum(0, keepdims=True)
        total = np.where(total == 0, 1, total)
        return x / total
    

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

class Environment(SimulationObject):
    def __init__(self):
        self._index = -1
        pass
    
    def temp_max(self):
        pass
    
    def temp_min(self):
        pass
    
    def photosynthetically_active_radiation(self):
        pass
    
    def relative_humidity(self):
        pass
    
    def wind_speed(self):
        pass
    
    def atmospheric_pressure(self):
        pass
    
    def ref_et(self):
        pass
    
    def rainfall(self):
        pass
    
    def maximum_kc(self):
        pass

class EnvironmentConstant(Environment):
    def __init__(self, tmax, tmin, rainfall, et0, kcmax):
        super(EnvironmentConstant, self)
        self._tmax = tmax
        self._tmin = tmin
        self._rainfall = rainfall
        self._et0 = et0
        self._kcmax = kcmax

class EnvironmentDataFrame(Environment):
    def __init__(self, data, repeat = False):
        super(EnvironmentDataFrame, self).__init__()
        self._data = data
        self.repeat = repeat
    
    def temp(self):
        return (self._data["tmin"][self._index] + self._data["tmax"][self._count]) / 2.
    
    def temp_max(self):
        return self._data["tmax"][self._index]
    
    def temp_min(self):
        return self._data["tmin"][self._index]
    
    def rh(self):
        return self._data["rh"][self._index]
    
    def rh_max(self):
        return self._data["rhmax"][self._index]
    
    def rh_min(self):
        return self._data["rhmin"][self._index]
    
    def wind_speed(self):
        return self._data["wind_speed"][self._index]
    
    def atmospheric_pressure(self):
        return self._data["atmospheric_pressure"][self._index]
    
    def ref_et(self):
        return self._data["ref_et"][self._index]
    
    def maximum_kc(self):
        kcb = self.simulation.kcb()
        h = self.simulation.crop_height()
        
        u2 = self.wind_speed()
        rhmin = self.rh_min()
        
        return maximum_kc(kcb, u2, rhmin, h)
    
    def rainfall(self):
        return self._data["rainfall"][self._index]
    
    def update(self):
        if self.repeat:
            self._index = (self._index + 1) % len(self._data)
        else:
            self._index += 1

class Simulation:
    def __init__(self, crop, soil, environment):
        self._crop = crop
        self._soil = soil
        self._environment = environment
        
        self._crop.simulation = self
        self._soil.simulation = self
        self._environment.simulation = self
        
        self._ev = None
        self._et = None
        
        self._day = -1
        self.initialize()
    
    def initialize(self):
        self._environment.initialize()
        self._crop.initialize()
        self._soil.initialize()
        
    def execute_step(self):
        self._day += 1
        
        self._environment.update()
        self._crop.update()
        self._soil.update()
    
    def _set_actual_evaporation(self, newvalue):
        self._ev = newvalue
    
    def _set_actual_transpiration(self, newvalue):
        self._et = newvalue
    
    def actual_evaporation(self):
        return self._ev
    
    def actual_transpiration(self):
        return self._et
    
    def potential_evaporation(self):
        ke = self.ke()
        et0 = self.et0()
        return ke * et0
    
    def potential_transpiration(self):
        kcb = self.kcb()
        et0 = self.ref_et()
        return kcb * et0
    
    def kcb(self):
        return self._crop.kcb()
        
    def crop_height(self):
        return self._crop.height()
    
    def rainfall(self):
        return self._environment.rainfall()
        
    def ref_et(self):
        return self._environment.ref_et()
    
    def f_ew(self):
        return 1 - self._crop.ground_covering()

    def irrigation(self):
        return 0
    
    def maximum_kc(self):
        return self._environment.maximum_kc()
    
    def ke(self):
        return self.maximum_kc() - self.kcb()

    def root_depth(self):
        return self._crop.root_depth()