import math
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.io import wavfile

from automations import ParamCurve



class ModulatedSignal():
  def __init__(self,
               amplitude_curve : ParamCurve ,
               frequency_curve : ParamCurve,
               phase_shift : float):
    assert isinstance(amplitude_curve, ParamCurve)
    assert isinstance(frequency_curve, ParamCurve)
    assert torch.equal(amplitude_curve.time_0, frequency_curve.time_0)
    assert torch.equal(amplitude_curve.delta_t, frequency_curve.delta_t)
    assert torch.equal(amplitude_curve.time_start, frequency_curve.time_start)
    assert torch.equal(amplitude_curve.time_end, frequency_curve.time_end)
    assert 0 <= phase_shift < 2 * np.pi

    self.amplitude_curve = amplitude_curve
    self.frequency_curve = frequency_curve

    self.t_0 = amplitude_curve.time_0
    self.delta_t = amplitude_curve.delta_t
    self.time_start = amplitude_curve.time_start
    self.time_end = amplitude_curve.time_end

    self.phase_shift = phase_shift

  
  def eval(self, t):
    amplitdue_factor = self.amplitude_curve.eval(t)
    frequency_factor = self.frequency_curve.eval(t)

    phase_term = self.phase_shift * torch.ones_like(t)

    angle = 2 * np.pi * frequency_factor * t + phase_term
    oscillator_value = np.sin(angle)

    signal_value = amplitdue_factor * oscillator_value

    return signal_value
  
  def save_wav(self, file_path : str, sample_rate : float) -> None:
    assert isinstance(file_path, str)
    assert len(file_path) > 3
    assert file_path[-3:-1] == '.wav'
    assert isinstance(sample_rate, float)
    assert sample_rate > 0.0

    sample_count = math.floor(2 * self.delta_t * sample_rate)
    step_size = 1/sample_rate
  
    time_sampling = [self.t_0 - self.delta_t + k * step_size for k in range(sample_count)]
    t = torch.Tensor(time_sampling)

    signal_value = self.eval(t)

    torchaudio.save(file_path, signal_value, sample_rate=sample_rate)
    print('Signal saved as `.wav` \'{}\' at sample rate {}.'.format(file_path, sample_rate))



def sample_parameters(number_of_samples : int, t_0 : float, delta_t : float, mean_frequency : float, sd_frequency : float) -> dict:
  assert isinstance(number_of_samples, int)
  assert number_of_samples > 0
  assert isinstance(t_0, float)
  assert isinstance(delta_t, float)
  assert delta_t > 0.0
  assert isinstance(mean_frequency, float)
  assert mean_frequency > 0.0
  assert isinstance(sd_frequency, float)
  assert sd_frequency > 0.0

  amp_time_mid = np.random.normal(t_0, delta_t/8., number_of_samples)
  amp_time_in = np.random.uniform(low=(t_0 - delta_t) * np.ones_like(amp_time_mid), high=amp_time_mid)
  amp_time_out = np.random.uniform(low=amp_time_mid, high=(t_0 + delta_t) * np.ones_like(amp_time_mid))
 
  freq_time_mid = np.random.uniform(low=t_0 - delta_t, high=t_0 + delta_t, size=number_of_samples)
  freq_time_in = np.random.uniform(low=(t_0 - delta_t) * np.ones_like(freq_time_mid), high=freq_time_mid)
  freq_time_out = np.random.uniform(low=freq_time_mid, high=(t_0 + delta_t) * np.ones_like(freq_time_mid))

  amplitude_mid = np.random.uniform(low=0.0, high=1.0, size=number_of_samples)
  amp_deg_in = np.exp(np.random.normal(2., np.log(2.), number_of_samples))
  amp_deg_out = np.exp(np.random.normal(0.5, np.log(2.), number_of_samples))

  frequency_mid = np.random.normal(mean_frequency, sd_frequency, number_of_samples)
  frequency_in = np.random.uniform(low=2/3, high=3/2, size=number_of_samples) * frequency_mid
  frequency_out = np.random.uniform(low=2/3, high=3/2, size=number_of_samples) * frequency_mid
  freq_deg_in = np.exp(np.random.normal(-np.log(4.0), np.log(4.0), number_of_samples))
  freq_deg_out = np.exp(np.random.normal(-np.log(4.0), np.log(4.0), number_of_samples))

  phase_shift = np.random.uniform(low=0.0, high=2 * np.pi, size=number_of_samples)

  signal_coeffs = np.random.uniform(low=0.0, high=1.0, size=number_of_samples)

  parameter_dictionary = {}
  for k in range(number_of_samples):
    parameter_dictionary[k] = {}
    parameter_dictionary[k]['time_interval'] = (t_0, delta_t)

    parameter_dictionary[k]['amp_events'] = (amp_time_in[k], amp_time_mid[k], amp_time_out[k])
    parameter_dictionary[k]['amplitude'] = (0.0, amplitude_mid[k], 0.0)
    parameter_dictionary[k]['amp_deg'] = (amp_deg_in[k], amp_deg_out[k])

    parameter_dictionary[k]['freq_events'] = (freq_time_in[k], freq_time_mid[k], freq_time_out[k])
    parameter_dictionary[k]['frequency'] = (frequency_in[k], frequency_mid[k], frequency_out[k])
    parameter_dictionary[k]['freq_deg'] = (freq_deg_in[k], freq_deg_out[k])

    parameter_dictionary[k]['phase_shift'] = phase_shift

    parameter_dictionary[k]['signal_coeff'] = signal_coeffs[k]

  return parameter_dictionary



class CompoundSignal():
  def __init__(self, 
               parameter_dictionary : dict):
    assert isinstance(parameter_dictionary, dict)
    assert len(parameter_dictionary) > 0
    
    self.t_0, self.delta_t = parameter_dictionary[0]['time_interval']

    self.signal_dict = {}
    for key, single_sample in parameter_dictionary.items():
      t_amp_in, t_amp_mid, t_amp_out = single_sample['amp_events']
      v_amp_in, v_amp_mid, v_amp_out = single_sample['amplitude']
      d_amp_in, d_amp_out = single_sample['amp_deg']
      # print(v_amp_in, v_amp_out)

      t_freq_in, t_freq_mid, t_freq_out = single_sample['freq_events']
      v_freq_in, v_freq_mid, v_freq_out = single_sample['frequency']
      d_freq_in, d_freq_out = single_sample['freq_deg']

      phase_shift = single_sample['signal_coeff']

      weight = single_sample['signal_coeff']

      amplitude_curve = ParamCurve(self.t_0, self.delta_t,
                                   v_amp_in, v_amp_mid, v_amp_out,
                                   t_amp_in, t_amp_mid, t_amp_out,
                                   d_amp_in, d_amp_out)

      frequency_curve = ParamCurve(self.t_0, self.delta_t,
                                   v_freq_in, v_freq_mid, v_freq_out,
                                   t_freq_in, t_freq_mid, t_freq_out,
                                   d_freq_in, d_freq_out)

      modulated_signal = ModulatedSignal(amplitude_curve, frequency_curve, phase_shift)

      self.signal_dict[key] = {}
      self.signal_dict[key]['weight'] = weight
      self.signal_dict[key]['amplitude_curve'] = amplitude_curve
      self.signal_dict[key]['frequency_curve'] = frequency_curve
      self.signal_dict[key]['modulated_signal'] = modulated_signal

  
  def eval(self, t):
    signal_value = torch.zeros_like(t)

    for key, component_signal_curves in self.signal_dict.items():
      component_modulated_signal = component_signal_curves['modulated_signal']
      component_amplitude_curve = self.signal_dict[key]['amplitude_curve']
      component_frequency_curve = self.signal_dict[key]['frequency_curve']
      weight = component_signal_curves['weight']

      component_signal_value = component_modulated_signal.eval(t)
      max_frequency = torch.max(component_frequency_curve.eval(t))
      freq_rel_weight = 1/max_frequency
      weighted_term = weight * freq_rel_weight  * component_signal_value

      signal_value = signal_value + weighted_term

    return signal_value
  

  def save_wav(self, file_path : str, sample_rate : int) -> None:
    assert isinstance(file_path, str)
    assert len(file_path) > 3
    assert file_path.split('.')[-1] == 'wav'
    assert isinstance(sample_rate, int)
    assert sample_rate > 0

    sample_count = math.floor(2 * self.delta_t * sample_rate)
    step_size = 1/sample_rate
  
    time_sampling = [self.t_0 - self.delta_t + k * step_size for k in range(sample_count)]
    t = torch.Tensor(time_sampling)
    t = t.to(dtype=torch.float32)

    signal_value = self.eval(t)
    scale_factor = torch.max(torch.abs(signal_value))
    signal_value = signal_value / (1.15 * scale_factor)

    audio_numpy = signal_value.numpy()
    audio_numpy_int32 = np.int32(audio_numpy * 2147483647)
    wavfile.write(file_path, sample_rate, audio_numpy_int32)
    print('Signal saved as \'{}\' at sample rate {}.'.format(file_path, sample_rate))




if __name__ == "__main__":
  step_size = 0.001
  step_count = math.floor((2 * 1.0) / step_size)
  input_points = [-1.0 + step_size * k for k in range(step_count)]

  param_dict = sample_parameters(16, 0., 1., 440., 330.)
  compound_signal = CompoundSignal(param_dict)

  compound_signal.save_wav('output.wav', 88200)
  
  tensor = compound_signal.eval(torch.Tensor(input_points))
  y_range = 1.15 * torch.max(torch.abs(tensor))
  tensor = tensor / y_range
  numpy_array = tensor.numpy()
  plt.plot(numpy_array)
  plt.xlabel('Index')
  plt.ylabel('Value')
  plt.title('1D Torch Tensor Plot')
  plt.xlim(0, len(numpy_array) - 1)
  plt.ylim(-1., 1.)
  plt.show()
  plt.clf()


  # amplitude_curve = ParamCurve(
  #   0.0, 0.5,
  #   0.0, 1.0, 0.0,
  #   -0.35, -0.15, 0.35,
  #   2.5, 2.5)

  # frequency_curve = ParamCurve(
  #   0.0, 0.5,
  #   440.0, 880.0, 220.0,
  #   -0.2, -0.1, 0.25,
  #   3.0, 0.5)

  # step_size = 0.001
  # step_count = math.floor((2 * amplitude_curve.delta_t) / step_size)
  # input_points = [amplitude_curve.time_start + step_size * k for k in range(step_count)]

  # signal = ModulatedSignal(amplitude_curve, frequency_curve, 0.)
  # tensor = signal.eval(torch.Tensor(input_points))
  # y_range = 1.15 * torch.max(torch.abs(tensor))
  # tensor = tensor / y_range

  # numpy_array = tensor.numpy()
  # plt.plot(numpy_array)
  # plt.xlabel('Index')
  # plt.ylabel('Value')
  # plt.title('1D Torch Tensor Plot')
  # plt.xlim(0, len(numpy_array) - 1)
  # plt.show()
  # plt.clf()