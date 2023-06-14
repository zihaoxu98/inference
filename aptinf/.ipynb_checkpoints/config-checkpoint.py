import typing as ty
import numpy as np
import inference_interface as ii

from copy import deepcopy
from immutabledict import immutabledict
from scipy.interpolate import RegularGridInterpolator

from aptinf.share import _cached_configs
from aptinf.utils import get_file_path

OMITTED = '<OMITTED>'


def takes_config(*configs):
    """Decorator for plugin classes, to specify which configs it takes.

    :param configs: Config instances of configs this plugin takes.
    """

    def wrapped(plugin_class):
        """
        :param plugin_class: plugin needs configuration
        """
        result = dict()
        for config in configs:
            if not isinstance(config, Config):
                raise RuntimeError("Specify config options by Config objects")
            config.taken_by = plugin_class.__name__
            result[config.name] = config

        if (hasattr(plugin_class, 'takes_config') and len(plugin_class.takes_config)):
            # Already have some configs set, e.g. because of subclassing
            # where both child and parent have a takes_config decorator
            for config in result.values():
                if config.name in plugin_class.takes_config:
                    raise RuntimeError(
                        f'Attempt to specify config {config.name} twice')
            plugin_class.takes_config = immutabledict({
                **plugin_class.takes_config, **result})
        else:
            plugin_class.takes_config = immutabledict(result)

        # Should set the configurations as the attributes of Plugin
        return plugin_class

    return wrapped


class Config():
    """Configuration option taken by a appletree plugin"""

    def __init__(self,
                 name: str,
                 type: ty.Union[type, tuple, list] = OMITTED,
                 default: ty.Any = OMITTED,
                 help: str = '',
                 **kwargs):
        """Initialization.

        :param name: name of the map
        :param type: Excepted type of the option's value.
        :param default: Default value the option takes.
        :param help: description of the map
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help

        # Sanity check
        if isinstance(self.default, dict):
            raise ValueError(
                f"Do not set {self.name}'s default value as dict!",
            )

    def get_default(self):
        """Get default value of configuration"""
        if self.default is not OMITTED:
            return self.default

        raise ValueError(f"Missing option {self.name} "
                         f"required by {self.taken_by}")

    def build(self):
        """Build configuration, set attributes to Config instance"""
        raise NotImplementedError


class Constant(Config):
    """Constant is a special config which takes only certain value"""

    value = None

    def __mul__(self, x):
        return x * self.value

    def __rmul__(self, x):
        return self.__mul__(x)

    def build(self):
        """Set value of Constant"""
        if self.name in _cached_configs:
            value = _cached_configs[self.name]
        else:
            value = self.get_default()
            # Update values to sharing dictionary
            _cached_configs[self.name] = value
        self.value = value


class Template(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hist_name = kwargs.get('hist_name', '')

    def __mul__(self, x):
        new_cls = deepcopy(self)
        new_cls.norm *= x
        return new_cls

    def __rmul__(self, x):
        return self.__mul__(x)

    def __imul__(self, x):
        self.norm *= x
        return self

    def build(self):
        if self.name in _cached_configs:
            self.file_path = _cached_configs[self.name]
        else:
            self.file_path = get_file_path(self.get_default())
            # Update values to sharing dictionary
            _cached_configs[self.name] = self.file_path

        self.hist = ii.template_to_multihist(self.file_path, hist_name=self.hist_name)
        self.norm = self.hist.histogram.sum()
        self._pdf = RegularGridInterpolator(self.hist.bin_centers(),
                                           self.hist.histogram / self.norm / self.hist.bin_volumes(), # BE CAREFUL
                                           bounds_error=False,
                                           fill_value=None)
        self.pdf = lambda x: np.clip(self._pdf(x), 1e-99, np.inf)
