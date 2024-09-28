import math
import numpy as np

from ..signals import CompoundSignal, sample_parameters


class StrikeSet():
  def __init__(self, sample_count, voice_count, t_0, delta_t, f_0, sigma_f):
    self.sample_count = sample_count
    self.count_width = math.ceil(np.log10(sample_count))

    self.t_0 = t_0
    self.delta_t = delta_t

    self.f_0 = f_0
    self.sigma_f = sigma_f

    self.voice_count = voice_count

    self.strike_dict = {}

  def save(self, file_name):
    for idx in range(self.sample_count):
      param_dict = sample_parameters(self.voice_count, self.t_0, self.delta_t, self.f_0, self.sigma_f)
      compound_signal = CompoundSignal(param_dict)

      padded_idx = str(idx).zfill(self.count_width)
      save_path = file_name + '_{}.wav'.format(padded_idx)
      
      compound_signal.save_wav(save_path, 88200)



if __name__ == "__main__":
  name_list = [32,6,0.,2.,880.,440.]
  strike_set = StrikeSet(*name_list)
  name_string = str(name_list).replace(' ', '').replace('.0','')
  strike_set.save('output_' + name_string)

