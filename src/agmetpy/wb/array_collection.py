import numpy as np

class ArrayCollection:
    def __init__(self, shape = (1,), **kwargs) -> None:
        self._arrays = kwargs
        self._shape = (1,)
        self._locked = False
        self._recalculate_shape()
        
    def _recalculate_shape(self):
        shapes = [self._shape]
        for i in self._arrays:
            shapes.append(np.shape(self._arrays[i]))
        shape = np.broadcast_shapes(*shapes)
        for i in self._arrays:
            self._arrays[i] = np.broadcast_to(self._arrays[i], shape)
            if np.shape(self._arrays[i]) != shape:
                self._arrays[i] = np.broadcast_to(self._arrays[i], shape)
        self._shape = shape
    
    def __getitem__(self, key) -> np.ndarray:
        return self._arrays[key]
    
    def __setitem__(self, key, value) -> None:
        self._arrays[key] = value
        if (self._shape != np.shape(value)):
            self._recalculate_shape()
    
    def __contains__(self, key) -> bool:
        return key in self._arrays
    
    def __str__(self) -> str:
        return f'ArrayCollection({self._arrays})'
    
    def __repr__(self) -> str:
        return str(self)
    
    def _get_shape(self) -> tuple:
        return self._shape
    
    shape = property(lambda self: self._get_shape())