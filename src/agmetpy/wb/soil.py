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

        dz_total = self.from_to(0, self._depth)

        self.ks = StressLinear(
            xmin=lambda: self.readily_available_water(dz_total),
            xmax=lambda: self.total_available_water(dz_total),
            x=lambda: self.depletion_from_fc(dz_total),
            reverse=True)
        
        self.kr = StressLinear(
            xmin=lambda: self.readily_evaporable_water(dz_total),
            xmax=lambda: self.total_evaporable_water(dz_total),
            x=lambda: self.depletion_from_fc(dz_total),
            reverse=True)
    
    def get_ks(self):
        return self._ks() if callable(self._ks) else self._ks
    
    def set_ks(self, value):
        self._ks = value

    ks = property(lambda self: self.get_ks(), lambda self, value: self.set_ks(value))

    def get_kr(self):
        return self._kr() if callable(self._kr) else self._kr
    
    def set_kr(self, value):
        self._kr = value
    
    kr = property(lambda self: self.get_kr(), lambda self, value: self.set_kr(value))
    
    def from_to(self, zmin, zmax):
        if np.any(zmin > zmax):
            raise Exception('zmin cannot be greater than zmax')
        if np.any(zmax > self._depth):
            raise Exception('zmax cannot be greater than Soil.depth')
        zini = np.broadcast_to(self._dz, self.soil_shape).cumsum(0) - self._dz
        return np.clip(zmax - zini, 0, self._dz) - np.clip(zmin - zini, 0, self._dz)
    
    def available_water(self, dz):
        return dz * np.maximum(self._theta - self._theta_wp, 0)
    
    def evaporable_water(self, dz):
        return dz * np.maximum(self._theta - 0.5*self._theta_wp, 0)

    def total_available_water(self, dz):
        return dz * (self._theta_fc - self._theta_wp)
    
    def readily_available_water(self, dz):
        return dz * (self._theta_fc - self._theta_wp) * self._p
    
    def total_evaporable_water(self, dz):
        return dz * (self._theta_fc - 0.5*self._theta_wp)
    
    def readily_evaporable_water(self, dz):
        return dz * (self._theta_fc - 0.5*self._theta_wp) * self._pe
    
    def depletion_from_sat(self, dz):
        return dz * np.maximum(self._theta_sat - self._theta, 0)

    def depletion_from_fc(self, dz):
        return dz * np.maximum(self._theta_fc - self._theta, 0)
    
    def depletion_from_wp(self, dz):
        return dz * np.maximum(self._theta_wp - self._theta, 0)
    
    def root_dist(self):
        zend = np.full(self.soil_shape, self._dz).cumsum(0)
        zini = zend - self._dz
        return self.simulation.crop.root_dist(zini, zend)
    
    def extract(self, depth, dist, theta_min):
        theta = np.copy(self._theta)
        theta_min = np.minimum(theta, theta_min)
        theta = np.maximum(theta - (depth * dist) / self._dz, theta_min)
        ext = (self._theta - theta) * self._dz
        self._theta = theta
        return ext
    
    def evaporate(self):
        pass

    def drainage_characteristic(self) -> np.ndarray:
        tau_day = np.minimum(0.0866 * (self._ksat * 8.64e7) ** 0.35, 1)
        return 1 - (1 - tau_day) ** (self.simulation.dt / 8.64e4)
    
    def update(self):
        rain = self.simulation.weather['rainfall']
        dtheta, dp, ro = self.drain(rain)

        kc_max = self.simulation.weather['kc_max']
        kcb = self.simulation.crop.kcb
        ks = self.ks
        few = self.simulation.management.few
        kr = self.kr
        ke = np.minimum(kr * (kc_max - kcb), few * kc_max)
        et_ref = self.simulation.weather['et_ref']

        dist = self.root_dist()

        etc = (kcb + ke) * et_ref
        et = (ks*kcb + ke) * et_ref

        
        # how it should be
        #
        # legend:
        # - few: exposed and wetted soil fraction
        # - fc: covered soil fraction
        # - (1-fc): exposed soil fraction
        # - fw: wetted soil fraction
        #
        # few = min(1 - fc, fw)
        # Ke = min(Kr * (Kcmax - Kcb), few * Kcmax)
        # ETc = (Ks * Kcb + Ke) * ET0

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
