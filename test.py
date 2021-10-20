#%%
import matplotlib
import matplotlib.pyplot as plt
import src.agmetpy.wb as wb
import numpy as np

th = np.repeat(0.2, 10)
fc = np.repeat(0.23, 10)
wp = np.repeat(0.1, 10)

rain = np.array([
    0, 0, 0, 10, 0, 0, 0, 0, 0, 0,
    0, 0, 20, 0, 0, 0, 0, 0, 0, 4
    ])

env = wb.Environment(tmax=32, tmin=24, rainfall=rain, rhmin=65, wind_speed=1.2, ref_et=5, repeat=True)
soil = wb.Soil(th, fc + 0.05, fc, wp, 300)
crop = wb.CropConstant(0.6, 0.5, 0.65, 0.8)

sim = wb.Simulation(crop, soil, env)

res_th = []
for i in range(1, rain.size):
    sim.execute_step()
    res_th.append(np.copy(sim._soil._theta))

res_th = np.stack(res_th, 0)

#%%
plt.imshow(res_th.transpose(), aspect='auto')
plt.show()
# %%
a, b, c = 1, 2, 3
a, b, c = np.atleast_1d(a, b, c)
a, b, c = np.broadcast_arrays(a, b, c)

# %%
