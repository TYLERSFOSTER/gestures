import generate


class Strikes():
  def __init__(self, count, t_0, delta_t):
    self.count = count
    self.t_0 = t_0
    self.delta_t = delta_t

    self.strike_dict = {}
    for idx in range(self.count):
      param_dict = generate.parameters(16, 0., 1., 440., 330.)
      compound_signal = generate.Strikes(param_dict)

      compound_signal.save_wav('output.wav', 88200)

