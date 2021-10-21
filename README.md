# AgmetPy

AgmetPy is a Python library aimed at agricultural forecasting, soil water balance, hidrology, and evapotranspiration research. It takes advantages of Numpy's vectorized and fast calculations.

Currently, AgmetPy has a complete set of functions for solar position calculations, atmospheric pressure and radiation balance over agricutural lands. It implements all the necessary functions to estimate reference evapotranspiration following the FAO's guidelines as described in FAO 56 (Allen et al., 1998) as well as classes for most of the soil water balance using the dual Kc approach.

We are working on functions to improve the methodology for soil water balance.

# Installing AgmetPy with pip

To install AgmetPy in your computer with Python 3 run:

```
pip install git+https://github.com/jvitorpinto/agmetpy.git
```

AgmetPy uses NumPy for calculations, which must be installed as well.

## References
ALLEN, R. G. et al. __Crop evapotranspiration: Guidelines for computing crop water requirements__. FAO Irrigation and drainage paper 56. Rome: Food and Agriculture Organization of the United Nations, 1998. Available from: <<https://www.fao.org/3/X0490E/x0490e00.htm>>. Access on: 18 Oct. 2021.
