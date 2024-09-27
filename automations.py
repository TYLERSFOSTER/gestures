import math
import numpy as np
import torch
import matplotlib.pyplot as plt


class ParamCurve():
  def __init__(self,
               t_0 : float, delta_t : float,
               v_in : float, v_mid : float, v_out : float,
               t_in : float, t_mid : float, t_out : float,
               d_in : float, d_out : float):
    assert isinstance(t_0, float)
    assert isinstance(delta_t, float)
    assert delta_t > 0.0
    assert isinstance(v_in, float)
    assert isinstance(v_mid, float)
    assert isinstance(v_out, float)
    assert isinstance(t_in, float)
    assert t_0 - delta_t <= t_in < t_0 + delta_t, 'The point `t_in` must lie in the interval [t_0-delta_t, t_0+delta_t]'
    assert isinstance(t_mid, float)
    assert t_0 - delta_t < t_mid < t_0 + delta_t, 'The point `t_mid` must lie in the interval [t_0-delta_t, t_0+delta_t]'
    assert isinstance(t_out, float)
    assert t_0 - delta_t < t_out <= t_0 + delta_t, 'The point `t_out` must lie in the interval [t_0-delta_t, t_0+delta_t]'
    assert t_in < t_mid < t_out, 'The points `t_in`, `t_mid`, and `t_out` must be linearly ordered.'
    assert isinstance(d_in, float)
    assert d_in > 0, 'Arching degree `d_in` must be a postive float value.'
    assert isinstance(d_out, float)
    assert d_out > 0, 'Arching degree `d_out` must be a postive float value.'
    
    self.time_0 = torch.Tensor((t_0,))
    self.delta_t = torch.Tensor((delta_t,))
    self.time_start = self.time_0 - self.delta_t
    self.time_end = self.time_0 + self.delta_t

    self.value_in = torch.Tensor((v_in,))
    self.value_mid = torch.Tensor((v_mid,))
    self.value_out = torch.Tensor((v_out,))

    self.time_in = torch.Tensor((t_in,))
    self.time_mid = torch.Tensor((t_mid,))
    self.time_out = torch.Tensor((t_out,))

    self.deg_in = torch.Tensor((d_in,))
    self.deg_out = torch.Tensor((d_out,))

    self.slope_in = \
      (self.value_mid - self.value_in) / \
        (self.time_mid - self.time_in)
    self.slope_out = \
      (self.value_out - self.value_mid) / \
        (self.time_out - self.time_mid)
    self.const_in = \
      (self.value_in * self.time_mid - self.value_mid * self.time_in) / \
        (self.time_mid - self.time_in)
    self.const_out = \
      (self.value_mid * self.time_out - self.value_out * self.time_mid) / \
        (self.time_out - self.time_mid)

  
  def linear_in(self, t):
    return self.slope_in * t + self.const_in
  

  def linear_out(self, t):
    return self.slope_out * t + self.const_out
  

  def arch_in(self, t):
    linear_in_value = self.linear_in(t)

    shifted_to_zero = \
      (linear_in_value - torch.min(self.value_in, self.value_mid)) / \
        torch.abs(self.value_mid - self.value_in)
    
    deformed = shifted_to_zero ** self.deg_in

    shifted_back = \
      deformed * torch.abs(self.value_mid - self.value_in) + torch.min(self.value_in, self.value_mid)
    
    return shifted_back
    

  def arch_out(self, t):
    linear_mid_value = self.linear_out(t)

    shifted_to_zero = \
      (linear_mid_value - torch.min(self.value_mid, self.value_out)) / \
        torch.abs(self.value_out - self.value_mid)
    
    deformed = shifted_to_zero ** self.deg_out

    shifted_back = \
      deformed * torch.abs(self.value_out - self.value_mid) + torch.min(self.value_mid, self.value_out)
    
    return shifted_back


  def eval(self, t : torch.Tensor) -> torch.Tensor:
    coeff_prehist = torch.heaviside(self.time_in - t, torch.Tensor([0.]))

    coeff_in = \
      torch.heaviside(t - self.time_in, torch.ones((1,))) * \
        torch.heaviside(self.time_mid - t, torch.zeros((1,)))
    
    coeff_out = \
      torch.heaviside(t - self.time_mid, torch.ones((1,))) * \
        torch.heaviside(self.time_out - t, torch.zeros((1,)))
    
    coeff_posthist = \
      torch.heaviside(t - self.time_out, torch.ones((1,)))

    term_prehist = coeff_prehist * self.value_in
    term_in = coeff_in * torch.nan_to_num(self.arch_in(t), nan=0.0)
    term_out = coeff_out * torch.nan_to_num(self.arch_out(t), nan=0.0)
    term_posthist = coeff_posthist * self.value_out

    curve_value = term_prehist + term_in + term_out + term_posthist

    return curve_value


if __name__ == "__main__":
  curve1 = ParamCurve(
    0.0, 0.5,
    -0.25, 1.0, 0.5,
    -0.15, 0.1, 0.2,
    3.0, 0.5)
  
  step_size = 0.001
  step_count = math.floor((2 * curve1.delta_t) / step_size)
  input_points = [curve1.time_start + step_size * k for k in range(step_count)]
  # print(input_points)
  tensor = curve1.eval(torch.Tensor(input_points))
  # print(tensor)
  numpy_array = tensor.numpy()
  
  plt.plot(numpy_array)
  plt.xlabel('Index')
  plt.ylabel('Value')
  plt.title('1D Torch Tensor Plot')
  plt.xlim(0, len(numpy_array) - 1)
  plt.ylim(-.5, 1.25)
  plt.show()
  plt.clf()

