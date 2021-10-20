import numpy as np
from numpy.core.fromnumeric import repeat
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
    def __init__(self, theta, theta_sat, theta_fc, theta_wp, ksat, dz = 0.1, ze = 0.1, p = 0.5):
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
        
        ksat : float, array_like
            Saturated hydraulic conductivity [mm/day].

        dz : float, array_like, optional
            Thickness of soil layers [m]. The default is 0.1.

        ze : float, array_like, optional
            Thickness of evaporation layer [m]. The default is 0.1.

        p : float, array_like, optional
            The FAO's coefficient for evapotranspiration reduction. The default is 0.5.
        '''

        theta, theta_sat, theta_fc, theta_wp, ksat = np.atleast_1d(
            theta, theta_sat, theta_fc, theta_wp, ksat
        )

        theta, theta_sat, theta_fc, theta_wp, ksat = np.broadcast_arrays(
            theta, theta_sat, theta_fc, theta_wp, ksat
        )

        # soil moisture [m³/m³]
        self._theta = theta
        # soil moisture at saturation [m³/m³]
        self._theta_sat = theta_sat
        # soil moisture at field capacity [m³/m³]
        self._theta_fc = theta_fc
        # soil moisture at wilting point [m³/m³]
        self._theta_wp = theta_wp
        # saturated hydraulic conductivity [mm/day]
        self._ksat = ksat
        # layer height [m]
        self._dz = dz
        # height of evaporative layer [m]
        self._ze = ze
        # number of layers and depth of soil [m]
        self.nlayers = theta.shape[0]
        self._depth = self.nlayers * dz
        # the FAO's p coefficient for evapotranspiration reduction.
        self._p = p
        # The fraction of the evaporable soil water that will be considered
        # as easily evaporable soil water.
        self._pe = 0.3
        self.option_calculate_drainage = True
    
    def initialize(self):
        # self._ks holds a reference to self._taw and self._raw, so both self._taw
        # and self._raw must be updated in place. The same occurs for self._kr.
        zr = self.simulation.root_depth()
        
        self._taw = np.atleast_1d(self.available_water(self._theta_fc, self._theta_wp, zr, total=True))
        self._raw = self._p * self._taw
        self._ks = stress.StressLinear(self._raw, self._taw, inverse=True)
        
        self._tew = np.atleast_1d(self.available_water(self._theta_fc, 0.5 * self._theta_wp, self._ze, total=True))
        self._rew = self._pe * self._tew
        self._kr = stress.StressLinear(self._rew, self._tew, inverse=True)

        self._tau = self.drainage_characteristic()
        # end initialize(self)
    
    def drainage_characteristic(self):
        return np.clip(0.0866 * self._ksat ** 0.35, 0, 1)

    def depletion_from(self, theta_ref, z=None, z0=0, total=False):
        '''
        Calculates the water depleted from a reference moisture between
        z0 and z. If total is set to True, returns the total depletion
        between z0 and z, otherwise, returns the depletion by layer
        between z0 and z.

        Params
        ------
        theta_ref: float, array_like
            The reference moisture.
        
        z: float, array_like

        z0: float, array_like
            The default is 0.
        
        total: boolean
            The default is false.
        '''
        de = np.fmax(theta_ref - self._theta, 0.) * self.from_to(z, z0)
        if total:
            de = de.sum(0)
        return de
    
    def calculate_drainage(self):
        #tau = np.broadcast_to(np.array([0.1]), self._theta.shape)
        tau = self._tau
        # variable to store the cumuluative drainage
        cumdr = np.zeros_like(np.atleast_1d(self._theta[0]))
        upper_da = 0
        for i in range(0, self.nlayers):
            # drainage ability, ie. how much theta varies per tick
            da = np.maximum(tau[i] * (self._theta[i] - self._theta_fc[i]), 0)
            # if the drainage ability of the current layer is smaller than
            # the drainage ability of upper layer, calculates how much moisture
            # the current layer must have in order to reach the same drainage
            # ability
            dcum = np.where(upper_da > da, self._dz * np.minimum(upper_da / tau[i], self._theta_sat[i]), 0)
            dcum = np.minimum(dcum, cumdr)
            # the cumulative drainage and the soil moisture are updated.
            cumdr = cumdr - dcum + da * self._dz
            self._theta[i] = self._theta[i] - da + dcum / self._dz
            upper_da = da
    
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
    
    def _update_taw(self, zr):
        taw = np.atleast_1d(self.available_water(self._theta_fc, self._theta_wp, zr, total=True))
        raw = self._p * self._taw
        
        np.copyto(self._taw, taw)
        np.copyto(self._raw, raw)
    
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
    
    def update(self):
        if (self.option_calculate_drainage):
            self.calculate_drainage()
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
        theta_min, theta_max:
            The interval of soil moisture where soil moisture
            will be depleted.
        z0, z: float
            The interval of depth from where the water will be
            depleted.
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
        weights = Soil.as_weights(aw)
        # the total extracted water is limited by the water available between z0 and z1
        ext_total = np.fmin(kr * depth, aw.sum(0))
        # then the extraction of water is distributed through layers by a weighting
        # factor calculated from the available water in each layer.
        ext = weights * ext_total
        self._theta -= ext / self._dz
        return depth - ext_total
    
    def _increase_theta_from_top(self, depth):
        # mudar isso depois
        dp = self.depletion_from_sat()
        #dp -= np.clip(depth - dp.cumsum(0), 0., dp)
        #self._theta = np.clip(self._theta_sat - dp / self._dz, 0, self._theta_sat)
        #dp = np.clip(depth - dp.cumsum(0), 0., dp)
        #self._theta = np.clip(self._theta + dp / self._dz, 0, self._theta_sat
        dr = depth
        for i in range(0, self.nlayers):
            ddepth = np.minimum(depth, dp[i])
            depth = depth - ddepth
            self._theta[i] = self._theta[i] + ddepth / self._dz
    
    def from_to(self, z=None, z0=0):
        '''
        Returns an array indicating how much of each layer
        is contained between z0 and z.

        For example, if a soil is made of 10 layers and each layer
        has a depth of 0.1m, them, calling this function with z0=0.17
        and z=0.52 will return the following array

        np.array([0.00, 0.03, 0.10, 0.10, 0.10, 0.02, 0.00, 0.00, 0.00, 0.00])

        Notice Δz = 0.52 - 0.17 = 0.35 and the sum of all numbers in the
        dimension 0 of the returned array is also 0.35.

        Parameters
        ----------
        z : array-like float, optional
            The second depth. If set to None, the total soil depth will used.

        z0 : array-like float, optional
            The first depth. The default is 0.

        Returns
        -------
        array-like float
            An array indicating how much of each layer is contained between
            z0 and z.

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
    def __init__(self, tmax, tmin, rainfall, rhmin, wind_speed, ref_et, repeat = False):
        tmax, tmin, rainfall, rhmin, wind_speed, ref_et = np.atleast_1d(
            tmax, tmin, rainfall, rhmin, wind_speed, ref_et)
        tmax, tmin, rainfall, rhmin, wind_speed, ref_et = np.broadcast_arrays(
            tmax, tmin, rainfall, rhmin, wind_speed, ref_et)
        
        self._index = -1
        self._length = np.size(tmax, 0)
        self._repeat = repeat
        self._tmax = tmax
        self._tmin = tmin
        self._rainfall = rainfall
        self._rhmin = rhmin
        self._wind_speed = wind_speed
        self._ref_et = ref_et
        
    def temp_max(self):
        return self._tmax[self._index]

    def temp_min(self):
        return self._tmin[self._index]
        
    def rh_min(self):
        return self._rhmin[self._index]

    def wind_speed(self):
        return self._wind_speed[self._index]

    def ref_et(self):
        return self._ref_et[self._index]

    def maximum_kc(self):
        kcb = self.simulation.kcb()
        h = self.simulation.crop_height()

        u2 = self.wind_speed()
        rhmin = self.rh_min()

        return maximum_kc(kcb, u2, rhmin, h)

    def rainfall(self):
        return self._rainfall[self._index]
        
    def update(self):
        if self._repeat:
            self._index = (self._index + 1) % self._length
        else:
            self._index += 1

class EnvironmentDataFrame(Environment):
    def __init__(self, data, repeat = False):
        tmax = data['tmax']
        tmin = data['tmin']
        rainfall = data['rainfall']
        rhmin = data['rhmin']
        wind_speed = data['wind_speed']
        ref_et = data['ref_et']
        super(EnvironmentDataFrame, self).__init__(tmax=tmax, tmin=tmin, rainfall=rainfall, rhmin=rhmin, wind_speed=wind_speed, ref_et=ref_et, repeat=repeat)

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
        ref_et = self.ref_et()
        return ke * ref_et
    
    def potential_transpiration(self):
        kcb = self.kcb()
        ref_et = self.ref_et()
        return kcb * ref_et
    
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