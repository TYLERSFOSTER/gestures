# __init__.py

from .automations import ParamCurve as AutoCurve
from .signals import ModulatedSignal as Signal
from .signals import CompoundSignal as Compound
from .signals import sample_parameters as parameters


__version__ = "1.0.0"
__author__ = "Tyler Foster"

def package_init_function():
    print("Package initialized")
