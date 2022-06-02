import src.agmetpy.wb as wb
import numpy as np

class ManagedCropConstant(wb.ManagedCrop, wb.CropConstant):
    pass

class ManagedCropNone(wb.ManagedCrop, wb.CropNone):
    pass

crop = src.agmetpy.wb.crop.CropManager(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]))