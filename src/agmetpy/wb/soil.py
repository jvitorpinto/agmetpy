import numpy as np

from .stress import StressLinear

from .simulation import SimulationObject

def _safe_divide(a, b, on_zero=np.inf):
    zerodiv = b == 0
    div = np.where(zerodiv, 1, b)
    return np.where(zerodiv, on_zero, a / div)

class Soil(SimulationObject):

    def __init__(
            self,
            theta: 'np.ndarray',
            theta_sat: 'np.ndarray',
            theta_fc: 'np.ndarray',
            theta_wp: 'np.ndarray',
            ksat: 'np.ndarray',
            dz: 'float' = 0.1,
            ze: 'float' = 0.1,
            p: 'float' = 0.5,
            pe: 'float' = 0.3,
            **kwargs):
        
        super(Soil, self).__init__(
            theta = theta,
            theta_sat = theta_sat,
            theta_fc = theta_fc,
            theta_wp = theta_wp,
            ksat = ksat,
            p = p,
            **kwargs)
        
        self._dz = dz
        self._ze = ze
        self._pe = pe

    def ks(self):
        '''
        Transpiration reduction coefficient.
        '''
        theta_raw = self._theta_wp + (1 - self._p) * (self._theta_fc - self._theta_wp)
        w = self._root_dist()
        return (np.clip((self._theta - self._theta_wp) / (theta_raw - self._theta_wp), 0, 1) * w).sum(0)
    
    def kr(self):
        '''
        Evaporation reduction coefficient.
        '''
        theta_rew = 0.5 * self._theta_wp + (1 - self._pe) * (self._theta_fc - 0.5 * self._theta_wp)
        w = self.evap_dist()
        return (np.clip((self._theta - 0.5 * self._theta_wp) / (theta_rew - 0.5 * self._theta_wp), 0, 1) * w).sum(0)
    
    def ke(self):
        '''
        Evaporation coefficient.
        '''
        kr = self.kr()
        kc_max = self._kc_max()
        kcb = self._kcb()
        few = self._few()
        ke1 = kr * (kc_max - kcb)
        ke2 = few * kc_max
        return np.minimum(ke1, ke2)
    
    def from_to(self, zmin, zmax):
        if np.any(zmin > zmax):
            raise Exception('zmin cannot be greater than zmax')
        if np.any(zmax > self._depth):
            raise Exception('zmax cannot be greater than Soil.depth')
        zini = np.broadcast_to(self._dz, self.soil_shape).cumsum(0) - self._dz
        return np.clip(zmax - zini, 0, self._dz) - np.clip(zmin - zini, 0, self._dz)
    
    def available_water(self):
        '''
        Calculates the water available in the root zone (i.e., water above
        wilting point).

        Returns
        -------
        A numpy.ndarray containing the water that roots can reach (m) for each cell
        of soil.
        '''
        return self._root_zone() * np.clip(self._theta - self._theta_wp, 0, self._theta_fc - self._theta_wp)
    
    def total_available_water(self):
        return self._root_zone() * (self._theta_fc - self._theta_wp)
    
    def readily_available_water(self):
        return self._root_zone() * (self._theta_fc - self._theta_wp) * self._p
    
    def evaporable_water(self):
        return self._ze * np.clip(self._theta - 0.5*self._theta_wp, 0, self._theta_fc - 0.5*self._theta_wp)

    def total_evaporable_water(self):
        return self._ze * (self._theta_fc - 0.5*self._theta_wp)
    
    def readily_evaporable_water(self):
        return self._ze * (self._theta_fc - 0.5*self._theta_wp) * self._pe
    
    def depletion_evap(self):
        return self._ze * np.maximum(self._theta_fc - self._theta, 0)
    
    def depletion_from_sat(self):
        return self._root_zone() * np.maximum(self._theta_sat - self._theta, 0)

    def depletion_from_fc(self):
        return self._root_zone() * np.maximum(self._theta_fc - self._theta, 0)
    
    def depletion_from_wp(self):
        return self._root_zone() * np.maximum(self._theta_wp - self._theta, 0)
    
    def _root_dist(self):
        '''
        Wrapper around Crop.root_dist, returns the root distribution for
        each layer of soil, or raise an Exception if a crop has not been
        set for self.simulation.crop.
        '''
        zend = np.full(self.soil_shape, self._dz).cumsum(0)
        zini = zend - self._dz
        return self.simulation.crop.root_dist(zini, zend)
    
    def _root_zone(self):
        return self.from_to(0, self.simulation.crop.zr)
    
    def drainage_characteristic(self) -> np.ndarray:
        tau_day = np.minimum(0.0866 * (self._ksat * 8.64e7) ** 0.35, 1)
        return 1 - (1 - tau_day) ** (self.simulation.dt / 8.64e4)

    def _kcb(self):
        return self.simulation.crop.kcb
    
    def _few(self):
        return self.simulation.management.few
    
    def _kc_max(self):
        return self.simulation.weather['kc_max']
    
    def _rainfall(self):
        return self.simulation.weather['rainfall']
    
    def _et_ref(self):
        return self.simulation.weather['et_ref']
    
    def evap_dist(self):
        return _safe_divide(self._ze, self._ze.sum(0), 0)

    def et_partitioning(self):
        # ke, ks, kcb and et_ref are the same shape as the layer shape.
        ke = self.ke()
        ks = self.ks()
        kcb = self._kcb()
        et_ref = self._et_ref()

        aw = self.available_water()
        ew = self.evaporable_water()

        # maximum crop evapotranspiration
        etc = (kcb + ke) * et_ref

        # actual evapotranspiration
        et = (ks*kcb + ke) * et_ref

        # actual evaporation and transpiration
        ev = np.minimum((ke * et_ref) * self.evap_dist(), ew)
        tr = np.minimum((et - ev) * self._root_dist(), aw)

        dtheta = (tr/self._dz) + (ev/self._dz)

        self._theta -= dtheta

        ret = {
            'delta_theta': dtheta,
            'ev': ev.sum(0),
            'tr': tr.sum(0),
            'tr_max': kcb * et_ref,
            'et_max': etc
        }

        return ret
    
    def update(self):
        rain = self._rainfall()
        dtheta1, dp, ro = self.drain(rain)
        dtheta2, ev, tr, tr_max, et_max = self.et_partitioning().values()

        ret = {
            'dp': dp,
            'ro': ro,
            'ev': ev,
            'tr': tr,
            'tr_max': tr_max,
            'et_max': et_max,
        }
    
    def drain(self, depth):
        tau = self.drainage_characteristic()
        theta = self._theta.copy()
        upper_layer_drainage_ability = np.full(self.layer_shape, np.inf)
        cumulative_drainage = np.copy(depth)
        excess = np.zeros(self.soil_shape).copy()

        for i in range(self.nlayers):
            drainage_ability = np.maximum(theta[i] - self._theta_fc[i], 0) * tau[i]
            theta[i] -= drainage_ability
            drainage_ability_after_drained = np.maximum(theta[i] - self._theta_fc[i], 0) * tau[i]
            theta_needed = np.where(
                drainage_ability_after_drained < upper_layer_drainage_ability,
                self._theta_fc[i] + np.minimum(_safe_divide(upper_layer_drainage_ability, tau[i]), self._theta_sat[i] - self._theta_fc[i]),
                theta[i])
            extraction_needed = (theta_needed - theta[i]) * self._dz
            extraction_possible = np.minimum(extraction_needed, cumulative_drainage)
            cumulative_drainage = cumulative_drainage - extraction_possible
            theta[i] += extraction_possible / self._dz
            cumulative_drainage = cumulative_drainage + drainage_ability * self._dz
            drainage_ability = np.maximum(theta[i] - self._theta_fc[i], 0) * tau[i]
            max_infiltration = self._ksat[i] * self.simulation.dt
            excess[i] = np.maximum(cumulative_drainage - max_infiltration, 0)
            cumulative_drainage = np.minimum(cumulative_drainage, max_infiltration)
            upper_layer_drainage_ability = drainage_ability
        
        depletion = (self._theta_sat - theta) * self._dz
        cumulative_excess = np.zeros(self.layer_shape)

        for i in range(self.nlayers):
            j = self.nlayers - i - 1
            cumulative_excess += excess[j]
            ext = np.minimum(depletion[j], cumulative_excess)
            theta[j] += ext
            cumulative_excess -= ext
        
        delta = (theta - self._theta) * self._dz
        self._theta = theta
        return delta, cumulative_drainage, cumulative_excess

    #-------------------------------------------------
    # theta_fc
    #-------------------------------------------------

    theta = property(
        lambda self: self._copy('theta'))
    
    _theta = property(
        lambda self: self._get('theta'),
        lambda self, value: self._set('theta', value))
    
    #-------------------------------------------------
    # theta_fc
    #-------------------------------------------------
    
    theta_fc = property(
        lambda self: self._copy('theta_fc'))
    
    _theta_fc = property(
        lambda self: self._get('theta_fc'),
        lambda self, value: self._set('theta_fc', value))
    
    #-------------------------------------------------
    # theta_sat
    #-------------------------------------------------
    
    theta_sat = property(
        lambda self: self._copy('theta_sat'))

    _theta_sat = property(
        lambda self: self._get('theta_sat'),
        lambda self, value: self._set('theta_sat', value))

    #-------------------------------------------------
    # theta_wp
    #-------------------------------------------------

    theta_wp = property(
        lambda self: self._copy('theta_wp'))

    _theta_wp = property(
        lambda self: self._get('theta_wp'),
        lambda self, value: self._set('theta_wp', value))
    
    #-------------------------------------------------
    # ksat
    #-------------------------------------------------

    ksat = property(lambda self: self._copy('ksat'))

    _ksat = property(
        lambda self: self._get('ksat'),
        lambda self, value: self._set('ksat', value))

    #-------------------------------------------------
    # p
    #-------------------------------------------------

    p = property(
        lambda self: self._copy('p'))

    _p = property(
        lambda self: self._get('p'),
        lambda self, value: self._set('p', value))
    
    #-------------------------------------------------
    # depth
    #-------------------------------------------------

    def get_depth(self):
        return self._var.shape[0] * self._dz
    
    depth = property(
        lambda self: self.get_depth())

    _depth = property(
        lambda self: self.get_depth())
    
    #-------------------------------------------------
    # nlayers
    #-------------------------------------------------

    def _get_nlayers(self):
        return self._var.shape[0]
    
    nlayers = property(
        lambda self: self._get_nlayers())
    
    _nlayers = property(
        lambda self: self._get_nlayers())
    
    #-------------------------------------------------
    # dz
    #-------------------------------------------------

    def _get_dz(self):
        return self._dz
    
    dz = property(
        lambda self: self._get_dz())
    
    #-------------------------------------------------
    # ze
    #-------------------------------------------------
    
    def _set_ze(self, value):
        self._set('ze', self.from_to(0, value))

    _ze = property(
        lambda self: self._get('ze'),
        lambda self, value: self._set_ze(value))

    ze = property(
        lambda self: self._copy('ze'),
        lambda self, value: self._set_ze(value))
    
    #-------------------------------------------------
    # soil_shape
    #-------------------------------------------------
    def _get_soil_shape(self):
        return self._var.shape

    soil_shape = property(
        lambda self: self._get_soil_shape())

    #-------------------------------------------------
    # layer_shape
    #-------------------------------------------------
    def _get_layer_shape(self):
        return self._var.shape[1:]
    
    layer_shape = property(
        lambda self: self._get_layer_shape())
