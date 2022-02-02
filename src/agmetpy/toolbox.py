# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 05:00:40 2021

@author: João Vitor de Nóvoa Pinto <jvitorpinto@gmail.com>
@version: 0.0.1
"""

import numpy as np
import enum

#-----------------------------------------------------------------------------
# Enum definitions
#-----------------------------------------------------------------------------

class ReferenceEvapotranspirationMethod(enum.Enum):
    FAO56 = 0
    ASCE_ET0 = 1
    ASCE_ETR = 2

class ClearnessIndexMethod(enum.Enum):
    FAO56 = 0
    ASCE = 1

#-----------------------------------------------------------------------------
# Support functions
#-----------------------------------------------------------------------------

def day_of_year(dt):
    """
    Calculates the day of year for a numpy.array of numpy.datetime64 date-time
    objects.
    
    Parameters
    ----------
    dt : numpy.datetime64
        Objects of type datetime64 representing the date for which the day of
        the year will be calculated.
    
    Returns
    -------
    int
        An integer between 1 and 366 representing the day of year for a given date.
    
    """
    return ((dt - dt.astype("datetime64[Y]")).astype("timedelta64[D]") + 1).astype(int)

def get_decimal_time(dt):
    return (dt - dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(int) / 3600

#-----------------------------------------------------------------------------

def _earth_sun_distance(j):
    x = j * 2 * np.pi / 365
    return 1 / (1 + 0.033 * np.cos(x))

def earth_sun_distance(dt):
    '''
    Calculates the distance between Earth and Sun on a given date in astronomical units.

    Parameters
    ----------
    dt : numpy.datetime64
        Date.

    Returns
    -------
    float
        Distance between Earth and Sun in astronomical units.

    '''
    j = day_of_year(dt)
    return _earth_sun_distance(j)

def _solar_declination(j):
    x = j * 2 * np.pi / 365
    return 0.409 * np.sin(x - 1.39)

def solar_declination(dt):
    '''
    Calculates solar declination, in radians. Positive values for North hemisphere
    summer and negative values during South hemisphere summer.

    Parameters
    ----------
    dt : numpy.datetime64
        Date.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    j = day_of_year(dt)
    return _solar_declination(j)

def _sunset_hour_angle(j, phi, delta):
    '''
    Calculates the sunset hour angle.

    Parameters
    ----------
    j : int
        Day of year ranging from 1 to 366.
    phi : float
        Latitude [rad]
    delta : float:
        Solar declination [rad]
    
    Returns
    -------
    float
        Sunset hour angle in radians (always positive).
    '''
    x = np.clip(-np.tan(phi) * np.tan(delta), -1, 1)
    return np.arccos(x)

def sunset_hour_angle(dt, lat):
    '''
    Calculates the sunset hour angle.
    
    Parameters
    ----------
    dt : numpy.datetime64
        Date.
    lat : float
        Latitute.
    
    Returns
    -------
    float
        Sunset hour angle in radians (always positive).
    
    '''
    j = day_of_year(dt)
    latrad = np.radians(lat)
    delta = _solar_declination(j)
    return _sunset_hour_angle(j, latrad, delta)

def _equation_of_time(j):
    b = (j - 81) * 2 * np.pi / 364
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)

def equation_of_time(dt):
    '''
    Calculates a correction for solar time in decimal hours.
    
    Parameters
    ----------
    dt : numpy.datetime64
        Date and time to calculate the correction to.
    
    Returns
    -------
    float
        Correction for solar time in decimal hours.
    '''
    j = day_of_year(dt)
    return _equation_of_time(j)

def solar_time_angle(dt, lon):
    '''
    Calculates solar time angle in radians.
    
    Solar time angle is negative in morning and positive in afternoon. At solar noon,
    the time angle is 0.
    
    Parameters
    ----------
    dt : numpy.datetime64
        Date at which the solar time angle must be calculated.
        
    lon : float
        Longitude.
    
    Returns
    -------
    float
        Solar time angle in radians.
    
    '''
    sc = equation_of_time(dt)
    x = np.mod(get_decimal_time(dt) + (lon / 15) + sc, 24)
    return (x - 12) * np.pi / 12

def daylight_hours(dt, lon):
    '''
    Calculates the number of hours with sunlight on a given date.
    
    Parameters
    ----------
    dt : numpy.datetime64
        Date.
    lon : float
        Longitude in decimal degrees.

    Returns
    -------
    float
        Number of hours with sunlight.
    
    '''
    omega_s = sunset_hour_angle(dt, lon)
    return omega_s * 24 / np.pi

def _daily_exoatmospheric_radiation(j, dr, phi, delta, omega):
    '''
    Calculates solar exoatmospheric radiation over a horizontal flat surface
    during a whole day.

    Parameters
    ----------
    j : int
        Day of year between 1-366.
    dr : float
        Inverse relative distance between Earth and Sun.
    phi : float
        Latitude [rad].
    delta : float
        Solar declination [rad].
    omega : float
        Sunset hour angle [rad].
    '''
    # 118022400 J/m² = 86400 s * 1366 W/m²
    k = 118022400 / np.pi
    return k * dr * (omega * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega))

def daily_exoatmospheric_radiation(dt, lat):
    '''
    Calculates solar exoatmospheric radiation over a horizontal flat surface
    during a whole day.
    
    Parameters
    ----------
    dt : numpy.datetime64
        Date as a numpy.datetime64 object.
    lat : float
        Latitude in decimal degrees.
    
    Returns
    -------
    float
        Solar exoatmospheric radiation [J/m²].
    
    '''
    j = day_of_year(dt)
    p = np.radians(lat)
    d = _solar_declination(j)
    o = _sunset_hour_angle(j, p, d)
    dr = 1 / _earth_sun_distance(j)
    return _daily_exoatmospheric_radiation(j, dr, p, d, o)

def hourly_exoatmospheric_radiation(dt_ini, dt_end, lat, lon):
    '''
    Calculates solar exoatmospheric radiation reaching a horizontal surface in
    a period smaller than a day.
    
    Even though the name of this function begins with 'hourly', it can calculate
    solar exoatmospheric radiation for time periods other than exactly 1 hour.
    
    Parameters
    ----------
    dt_ini : numpy.datetime64
        Initial time.
    dt_end : numpy.datetime64
        End time.
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    
    Returns
    -------
    float
        Solar exoatmospheric radiation over a horizontal flat surface (J/m²)

    '''
    # intermediary time is calculated as initial time (dt_ini) plus half way
    # the difference between final time (dt_end) and initial time (dt_ini)
    half_timedelta = (dt_end - dt_ini) / 2
    dt_mid = dt_ini + half_timedelta
    
    time_angle_mid = solar_time_angle(dt_mid, lon)
    
    time_angle_sunset = sunset_hour_angle(dt_mid, lat)
    dr = 1 / earth_sun_distance(dt_mid)
    decl = solar_declination(dt_mid)
    
    delta_time_angle = np.pi * (dt_end - dt_ini).astype("timedelta64[s]").astype(int) / 86400
    
    time_angle_ini = time_angle_mid - delta_time_angle
    time_angle_end = time_angle_mid + delta_time_angle
    
    def ra(dr, omega_ini, omega_end, phi, delta):
        k = 59011200 # 59011200 J/m² = 43200 s * 1366 W/m²
        p1 = (omega_end - omega_ini) * np.sin(phi) * np.sin(delta)
        p2 = np.cos(phi) * np.cos(delta) * (np.sin(omega_end) - np.sin(omega_ini))
        
        return k * dr * (p1 + p2) / np.pi
    
    #-------------------------------------------------------------------------
    
    # This procedure is necessary to make the sum of the exoatmospheric radiation
    # throughout a whole day equal to the result of the function
    # daily_exoatmospheric_radiation
    
    # time_angle_***_corr is the time angle adjusted to the range [-pi, +pi]
    time_angle_ini_corr = np.mod(time_angle_ini + np.pi, 2 * np.pi) - np.pi
    time_angle_end_corr = np.mod(time_angle_end + np.pi, 2 * np.pi) - np.pi
    
    # when the initial time angle lies before local midnight and the final time
    # angle lies after the local midnight and the actual night period is short
    # enough to fall entirely between time_angle_ini and time_angle_end, a
    # error will occur in the calculation of the exoatmospheric radiation, because
    # this algorithm calculates only the solar radiation reaching the surface
    # betweem time_angle_ini and time_angle_sunset or between -time_angle_sunset and
    # time_angle_end. Here the condition that leads to this error is verified
    # in order to further correct it.
    tini_sun = np.logical_and(time_angle_ini_corr > - time_angle_sunset,
                              time_angle_ini_corr < time_angle_sunset)
    tend_sun = np.logical_and(time_angle_end_corr > - time_angle_sunset,
                              time_angle_end_corr < time_angle_sunset)
    is_split = time_angle_end_corr < time_angle_ini_corr
    
    # error_condition is true when the above metioned error happens.
    error_condition = tini_sun & tend_sun & is_split
    
    corr_ini = np.where(error_condition, time_angle_sunset, 0)
    corr_end = np.where(error_condition, 2 * np.pi - time_angle_sunset, 0)
    
    corr = ra(dr, corr_ini, corr_end, np.radians(lat), decl)
    
    #-------------------------------------------------------------------------
    
    time_angle_ini = np.where(error_condition, time_angle_ini, np.clip(time_angle_ini, -time_angle_sunset, time_angle_sunset))
    time_angle_end = np.where(error_condition, time_angle_end, np.clip(time_angle_end, -time_angle_sunset, time_angle_sunset))
    
    erad = ra(dr, time_angle_ini, time_angle_end, np.radians(lat), decl)
    
    return  erad - corr

def solar_elevation_angle(dt, lat, lon):
    '''
    Calculates solar elevation angle for a given moment and location on the
    Earth's surface.
    
    Details
    -------
    Solar zenith angle (α) is calculated as
    
    sin(α) = cos(ϕ) * cos(δ) * cos(ω) + sin(ϕ) * sin(δ)
    
    where:
        ϕ - latitude [rad]
        δ - solar declination [rad]
        ω - solar time angle [rad]
    
    Parameters
    ----------
    dt : numpy.datetime64
        Moment at which solar elevation angle will be calculated.
    lat : float
        Local latitude in decimal degrees.
    lon : float
        Local longitude in decimal degrees.
    
    Returns
    -------
    float
        Solar elevation angle in radians.
    '''
    hour_angle = solar_time_angle(dt, lon)
    phi = np.radians(lat)
    declination = solar_declination(dt)
    return np.arcsin((np.cos(phi) * np.cos(declination) * np.cos(hour_angle)) + (np.sin(phi) * np.sin(declination)))

def solar_zenith_angle(dt, lat, lon):
    '''
    Calculates solar zenith angle for a given moment and location on the
    Earth's surface.
    
    Solar zenith angle (Φ) is calculated as
    
    cos(Φ) = cos(ϕ) * cos(δ) * cos(ω) + sin(ϕ) * sin(δ)
    
    where:
        ϕ - latitude [rad]
        δ - solar declination [rad]
        ω - solar time angle [rad]
    
    Parameters
    ----------
    dt : numpy.datetime64
        Moment.
    lat : float
        Decimal latitude in degrees.
    lon : float
        Decimal longitude in degrees.

    Returns
    -------
    float
        Solar zenith angle in radians.

    '''
    hour_angle = solar_time_angle(dt, lon)
    phi = np.radians(lat)
    declination = solar_declination(dt)
    return np.arccos((np.cos(phi) * np.cos(declination) * np.cos(hour_angle)) + (np.sin(phi) * np.sin(declination)))

def instantaneous_exoatmospheric_irradiance(dt, lat, lon):
    '''
    Calculates the exoatmospheric solar irradiance over a horizontal surface at a
    given moment.
    
    The exoatmospheric irradiance is given by
    Gsc * 1 / dr * cos(Φ)
    where:
        Gsc: solar constant (1366 W/m²)
        dr: relative earth sun distance
        Φ: solar zenith angle
    
    Parameters
    ----------
    dt : numpy.datetime64
        Date and time UTC.
    lat : float
        Latitude (decimal degrees).
    lon : float
        Longitude (decimal degrees).

    Returns
    -------
    float
        instantaneous_exoatmospheric_irradiance [W/m²].

    '''
    dr = 1 / earth_sun_distance(dt)
    sz = solar_zenith_angle(dt, lat, lon)
    return np.fmax(1366 * dr * np.cos(sz), 0)

def net_shortwave_radiation(rs, albedo = 0.23):
    '''
    Calculates net shortwave radiation as the difference between incoming
    shortwave radiation and reflected shortwave radiation on a horizontal
    surface.
    
    Parameters
    ----------
    rs : float
        Incoming shortwave radiation on a horizontal surface [energy / time / area].
    albedo : float
        Albedo of the horizontal surface [adimensional]. The default value is 0.23,
        which is the albedo of the reference crop for calculation of reference
        evapotranspiration by means of the standardized Penman-Monteith FAO
        equation.
    
    Returns
    -------
    float
        The difference between incoming and reflected shortwave radiation (same unit as rs).
    
    '''
    return (1 - albedo) * rs

def cloud_function(rs, rs0, se):
    '''
    Calculates the cloud-dependant part of the equation for net
    longwave radiation.

    Parameters
    ----------
    rs : float
        Solar global irradiance over a horizontal surface [J / m²]
    rs0 : float
        Solar global irradiance over a horizontal surface reaching the top of atmosphere [J / m²]
    se : float
        Solar elevation angle [rad]
    '''
    rr = np.clip(rs / np.where(se > 0.3, rs0, 1), 0.3, 1)
    return 1.35 * rr - 0.35

def net_longwave_radiation(tmax, tmin, ea, cf = None, hourly = False):
    '''
    Calculates net longwave radiation for daily or hourly periods.
    
    Parameters
    ----------
    tmax : float
        Maximum air temperature [K].
    tmin : float
        Minimum air temperature [K].
    ea : float
        Water vapour pressure [Pa].
    rs : float
        Solar radiation on a horizontal surface [J/m²].
    rs0 : float
        Solar radiation on a horizontal surface in clear sky conditions [J/m²].
    se : float, optional
        Solar elevation angle at the midpoint of the calculation period.
        Not necessary if hourly is set False. The default is None.
    cf_pred : float, optional
        DESCRIPTION. The default is None.
    se_threshold : float, optional
        DESCRIPTION. The default is 0.3.
    hourly : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    float
        net longwave radiation [J/m**2].

    '''
    
    # p1 - emission of longwave radiation by air
    # p2 - effect of water vapour
    # cf - effect of cloudness
    mult = 3600 if (hourly) else 86400
    p1 = mult * (stefan_boltzmann_law(tmax) + stefan_boltzmann_law(tmin)) / 2
    p2 = 0.34 - 0.004427188724235732 * np.sqrt(ea)
    
    return -(p1 * p2 * cf)

def clear_sky_radiation(z, ra):
    return (0.75 + 2e-5 * z) * ra

def latent_heat_of_vaporization(temp):
    '''
    Calculates the latent heat of vaporization of water as a function of
    temperature.

    Parameters
    ----------
    temp : float
        Temperature [K].

    Returns
    -------
    float
        latent heat of vaporization of water [J/kg].

    '''
    return 3145907.15 - 2361 * temp

def atmospheric_pressure(z, temp = 293.15, lb = 6.5e-3):
    """
    Calculates atmospheric pressure at a given height.

    Parameters
    ----------
    z : float
        Altitude above sea level [m].
    temp : float, optional
        Meam atmospheric temperature [K]. The default is 288.15.
    lb : float, optional
        Temperature lapse rate [K / m] (i.e. how many Kelvin the temperature of air decreases
        with a 1 m increase in altitude). The default is 5e-3 K / m.
    
    Returns
    -------
    float
        Atmospheric pressure at altitude z.

    """
    p0, g, rd = 101325, 9.80665, 287.058
    power = -g / (rd * lb)
    return p0 * ((temp + lb * z) / temp) ** power

def log_wind_profile(u1, z2 = 2, z1 = 10, d = 0.084, z0 = 0.01476):
    """
    Estimates wind speed at height z2 based on wind speed measued at
    height z1, given the height of the zero plane displacement and the roughness
    length of the surface.
    
    For a standardized FAO Peman Monteith ET0 surface
    you can use the default values for d and z0. If the wind speed is measured
    at a standard weather station, which measures wind speed
    at a 10m height, you can use the default value for z1.
    
    Parameters
    ----------
    u1 : float
        Wind speed [m/s] measured at height z1.
    z2 : float
        Height z2 [m].
    z1 : float, optional
        Height z1 [m]. The default is 10.
    d : float, optional
        Zero plane displacement height. If not set, a default value = 0.08 will be set, which
        is the zero plane displacement height estimated for a 0.12m height uniform crop.
    z0 : float, optional
        Roughness length. If not set, a default value of 0.01476 will be set, which
        corresponds to the roughness length of the standardized FAO ET0 Penman-Monteith
        equation.
    
    Returns
    -------
    float
        Wind speed at height z2.

    """
    return u1 * np.log((z2 - d) / z0) / np.log((z1 - d) / z0)

def air_density(temp, patm, pw = 0):
    """
    Calculates the density of dry air by means of the universal gas law as a
    function of air temperature and atmospheric pressure.
    
    m / V = [Pw / (Rv * T)] + [Pd / (Rd * T)]
    
    where:
        Pd: Patm - Pw
        Rw: specific gas constant for water vapour [Rw = 461.495 MJ/kg/K]
        Rv: specific gas constant for dry air [Rv = 287.058 MJ/kg/K]
        T: air temperature [K]
        m/V: density of air [kg/m³]
    
    Parameters
    ----------
    temp : float
        Air temperature [K].
    patm : float
        Atmospheric pressure [Pa].
    pw : float
        Vapour pressure [Pa]. Default to 0 Pa (dry air).
    
    Returns
    -------
    float
        Air density [kg/m³].

    """
    rd, rw = 287.058, 461.495 # specific gas constant for dry air and water vapour [J / (kg K)]
    pd = patm - pw
    return (pd / (rd * temp)) + (pw / (rw * temp))

def absolute_humidity(ea, temp):
    '''
    Calculates absolute humidity from partial pressure of water vapour and air
    temperature.
    
    Parameters
    ----------
    ea : float
        Partial pressure of water vapour [Pa].
    temp : float
        Absolute temperature [K].
    
    Returns
    -------
    float
        Absolute humidity [kg / m**3].
    
    '''
    rw = 461.495 # specific gas constant for water vapour [J / (kg K)]
    return ea / (rw * temp)

def specific_humidity(patm, ea, temp):
    '''
    Calculates specific humidity of air.
    
    Parameters
    ----------
    patm : float
        Atmospheric pressure [Pa].
    ea : float
        Partial pressure of water vapour [Pa].
    temp : float
        Absolute temperature [K].

    Returns
    -------
    float
        Specific humidity [kg / kg].

    '''
    rho = air_density(temp, patm, pw = ea)
    hum = absolute_humidity(ea, temp)
    return hum / rho

def vapour_pressure_from_absolute_humidity(h, temp):
    """
    Calculates the partial pressure of water vapour for a given absolute air
    humidity and temperature.

    Parameters
    ----------
    h : float
        absolute humidity [kg/m³].
    temp : float
        air temperature [K].

    Returns
    -------
    float
        Partial pressure of water vapour [Pa].

    """
    rw = 461.495 # specific gas constant for water vapour [J / (kg K)]
    return h * temp * rw

def vapour_pressure_from_specific_humidity(q, patm, temp, max_iter = 20):
    """
    Returns the partial pressure of water vapour for a given specific air
    humidity and atmospheric condition.
    
    Parameters
    ----------
    q : float
        specific humidity of air [kg/kg].
    patm : float
        atmospheric pressure [Pa].
    temp : float
        air temperature [K].
    max_iter : integer, optional
        max number of iterations until it stops. The default is 20.
    
    Returns
    -------
    pw : float
        Vapour pressure [Pa] for a given specific air humidity.
    
    """
    rw = 461.495 # specific gas constant for water vapour [J / (kg K)]
    pw = 0
    stop = False
    i = 0
    
    while i < max_iter or (not np.all(stop)):
        rho = air_density(temp, patm, pw)
        new_pw = rho * q * rw * temp
        stop = new_pw == pw
        pw = new_pw
        i += 1
    
    return pw

def vapour_saturation_pressure(temp):
    '''
    Calculates the vapour saturation pressure. Vapour saturation pressure is
    the partial pressure of water vapour mixed to the air when the air is at
    saturation.
    
    Parameters
    ----------
    temp : float
        Absolute temperature of air [K].

    Returns
    -------
    float
        vapour saturation pressure [Pa].

    '''
    return 610.78 * np.exp((17.27 * temp - 4717.3005) / (temp - 35.85))

def coef_direct_beam_radiation(patm, ea, sz, kt, method = ClearnessIndexMethod.ASCE):
    '''
    Calculates the coefficient for direct beam radiation for estimation of
    atmosphere transparency.
    
    Parameters
    ----------
    patm : float
        atmospheric pressure [Pa].
    ea : float
        partial pressure of water vapour [Pa].
    sz : float
        solar zenith angle [rad].
    kt : float
        clearness index [adimensional].
    method : ClearnessIndexMethod, optional
        One of ClearnessIndexMethod.FAO56 or ClearnessIndexMethod.ASCE.
        The default is ClearnessIndexMethod.ASCE.
    
    Returns
    -------
    float
        Coefficient for direct beam radiation.

    '''
    isnight = sz >= np.pi / 2
    
    cossz = np.where(isnight, 1, np.cos(sz))
    w = 0.14e-6 * patm * ea + 2.1
    
    if method == ClearnessIndexMethod.FAO56:
        c1, c2 = 0.091, 0.25
    elif method == ClearnessIndexMethod.ASCE:
        c1, c2 = 0.075, 0.40
    
    return np.where(isnight, 0, 0.98 * np.exp((-1.46e-6 * patm / (kt * cossz)) - (c1 * (w / cossz) ** c2)))

def coef_diffuse_beam_radiation(kb, method = ClearnessIndexMethod.ASCE):
    '''
    Calculates the coefficient for diffuse beam radiation.
    
    Parameters
    ----------
    kb : float
        Coefficient for direct beam radiation, which can be calculated by
        coef_direct_beam_radiation.
    method : ClearnessIndexMethod, optional
        One of ClearnessIndexMethod.FAO56 or ClearnessIndexMethod.ASCE.
        The default is ClearnessIndexMethod.ASCE.
    
    Returns
    -------
    float
        Coefficient for diffuse beam radiation.
        
    '''
    if method == ClearnessIndexMethod.FAO56:
        c1, c2, c3, c4 = 0.35, -0.33, 0.18, 0.82
    elif method == ClearnessIndexMethod.ASCE:
        c1, c2, c3, c4 = 0.35, -0.36, 0.18, 0.82
    
    return np.where(kb >= 0.15, c1 + c2 * kb, c3 + c4 * kb)

def clearness_index(patm, ea, sz, kt = 1, method = ClearnessIndexMethod.ASCE):
    '''
    Calculates the fraction of exoatmospheric radiation that reaches Earths's
    surface.
    
    Parameters
    ----------
    patm : float
        Atmospheric pressure [Pa].
    ea : float
        Vapour pressure [Pa].
    sz : float
        Solar zenith angle [radians]
    kt : float, optional
        Turbidity coefficient [adimensional]. kt = 1 for clear atmosphere and 0.5 <= kt < 1 for polluted atmosphere.
        The default is 1.
    
    Returns
    -------
    float
        clearness index, adimensional, between 0 and 1.
    
    '''
    kb = coef_direct_beam_radiation(patm, ea, sz, kt, method)
    kd = coef_diffuse_beam_radiation(kb, method)
    return kb + kd

def solar_incidence_angle(delta, phi, omega, slope = 0, aspect = 0):
    '''
    Calculates the solar incidence angle over a non horizontal surface given the
    surface's slope and orientation as well as the local solar position. If
    slope = 0 then the solar incidence angle is equal to local solar zenith.
    
    Parameters
    ----------
    delta : float
        Solar declination angle, in radians, positive during the summer of the
        north hemisphere and negative during the summer of the south hemisphere.
        
    phi : float
        Local latitude, in radians. Positive for locations in the northern hemisphere
        and negative for locations in the southern hemisphere.
        
    omega : float
        Solar time angle, in radians, negative during morning and positive on afternoon.
        
    slope : float, optional
        Slope of surface, where 0 means a horizontal surface and pi / 2 a vertical surface. The default is 0.
        
    aspect : float, optional
        Orientation of the surface, where aspect = 0 if the surface faces southward. The default is 0.
    
    Returns
    -------
    float
        solar incidence angle over the surface [rad].

    '''
    
    # first character:
    #   s - sin
    #   c - cos
    # second character:
    #   d - solar declination
    #   p - latitude in radians
    #   o - time angle
    #   s - surface slope
    #   a - surface aspect
    
    sd, cd = np.sin(delta), np.cos(delta)
    sp, cp = np.sin(phi), np.cos(phi)
    so, co = np.sin(omega), np.cos(omega)
    ss, cs = np.sin(slope), np.cos(slope)
    sa, ca = np.sin(aspect), np.cos(aspect)
    
    p1 = sd * sp * cs
    p2 = sd * cp * ss * ca
    p3 = cd * cp * cs * co
    p4 = cd * sp * ss * ca * co
    p5 = cd * sa * ss * so
    return np.arccos(p1 + p2 + p3 + p4 + p5)

def stefan_boltzmann_law(temp, epsilon = 1):
    '''
    Calculates the power radiated by a black body. Optionally, calculates
    the power radiated by a body given its emissivity ε < 1.

    Parameters
    ----------
    temp : float
        Body's absolute temperature [K].
    epsilon : float, optional
        Body's emissivity adimensional between 0 and 1. The default is 1.
        
    Returns
    -------
    float
        Power radiated by the body [W/m**2].

    '''
    sigma = 5.670374419e-8 # stefan-boltzmann constant [W/K**4/m**2]
    return epsilon * sigma * temp ** 4

def growing_degree_days(tb, temp_mean):
    return np.maximum(temp_mean - tb, 0)

def growing_degree_days(lower_tb, upper_tb, tmin, tmax):
    pass


def reference_evapotranspiration(tmax, tmin, rn, g, psy, u2, es, ea, method = ReferenceEvapotranspirationMethod.ASCE_ET0, hourly = False):
    '''
    Calculates the reference evapotranspiration for short or tall reference surface
    by means of the standardized Penman Monteith equation.
    
    BE AWARE This function does not use standard SI units, so parameters calculated
    by other functions in this module must be converted to the right units before
    calculations.
    
    Parameters
    ----------
    tmax : float
        Maximum air temperature [°C].
    tmin : float
        Minimum air temperature [°C].
    rn : float
        Net radiation [MJ/m²].
    g : float
        Soil heat flux [MJ/m²].
    psy : float
        Psychrometric constant [kPa/°C].
    u2 : float
        Wind speed measured at 2 m height [m/s].
    es : float
        Vapour saturation pressure [kPa].
    ea : float
        Vapour pressure [kPa].
    method : agmet.ReferenceEvapotranspirationMethod, optional
        Calculation method (see details). The default is ReferenceEvapotranspirationMethod.ASCE_ET0.
    hourly : TYPE, optional
        DESCRIPTION. The default is False.
    
    Details
    -------
    By now, the parameter 'method' can take one of three possibilites:
        
    - agmet.ReferenceEvapotranspirationMethod.FAO56:
        uses the standard FAO 56 coefficients.
        
    - agmet.ReferenceEvapotranspirationMethod.ASCE_ET0:
        uses the coefficients defined for the short reference surface (clipped grass, 0.12 m height)
        according to the American Society of Civil Engineers.
    
    - agmet.ReferenceEvapotranspirationMethod.ASCE_ETR:
        uses the coefficients defined for the tall reference surface (alfafa, 0.5 m height)
        according to the American Society of Civil Engineers.
    
    FAO56 and ASCE_ET0 are equal for 24h calculations of ET0, but they differ slightly
    for hourly calculations.
    
    Returns
    -------
    float
        Reference evapotranspiration [mm / day] or [mm / hour].

    '''
    temp = (tmin + tmax) / 2.
    
    slope = (4098. * vapour_saturation_pressure(temp + 273.15) / 1000.) / (237.3 + temp) ** 2
    
    # sets the coefficients cn and cd
    if (method == ReferenceEvapotranspirationMethod.FAO56):
        cn, cd = (37., 0.34) if (hourly) else (900., 0.34)
    elif (method == ReferenceEvapotranspirationMethod.ASCE_ET0):
        cn, cd = (37., np.where(rn > 0, 0.24, 0.96)) if hourly else (900., 0.34)
    elif (method == ReferenceEvapotranspirationMethod.ASCE_ETR):
        cn, cd = (66., np.where(rn > 0, 0.25, 1.7)) if hourly else (1600., 0.38)
    
    p1 = 0.408 * slope * (rn - g)
    p2 = psy * cn * u2 * (es - ea) / (temp + 273.)
    p3 = slope + psy * (1. + cd * u2)
    return (p1 + p2) / p3