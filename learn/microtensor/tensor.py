
import numpy as np

class Tensor:
  def __init__(self, data):
    self._data = data

  def data(self):
    return self._data

  def sum(self):
    sum = 0
    for v in np.ndarray.flatten(self._data):
      sum += v
    return sum

  def __str__(self):
	  return "hello world"
