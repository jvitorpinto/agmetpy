
#%%
import src.agmetpy.wb as wb
import numpy as np
import matplotlib.pyplot as plt

tmax = np.array(3 * [33, 32, 35, 30, 32, 31, 33, 29, 31, 34])
rain = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 200, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]) / 1000

c = wb.CropConstant(0.8, 0.5, 0.5, 0.8)
s = wb.Soil([[0.3 - i*0.01] for i in range(10)], 0.3, 0.2, 0.1, 0.1/86400, 0.1)
w = wb.Weather(temp_max=tmax, temp_min=22, rainfall=rain)

sim = wb.Simulation(c, s, w)

dist = np.maximum([[0.4 - i*0.1] for i in range(10)], 0)

theta = [sim.soil.theta]
ro = [[0]]
dp = [[0]]
for i in sim:
    delta, down, up = sim.soil.drain(sim.weather['rainfall'])
    ro.append(up)
    dp.append(down)
    theta.append(sim.soil.theta)

theta = np.stack(theta, 1)
ro = np.stack(ro, 1)[0]
dp = np.stack(dp, 1)[0]

plt.figure(0)
plt.imshow(theta,extent=(0, 31, 1, 0), aspect='auto')
plt.figure(1)
plt.plot(1000*dp)
plt.plot(1000*ro)

# %%

def safe_divide(a, b, on_zero=np.inf):
    zerodiv = b == 0
    div = np.where(zerodiv, 1, b)
    return np.where(zerodiv, on_zero, a / div)
    

safe_divide(np.array([1, 2, 3]), np.array([0, 1, 2]))
# %%