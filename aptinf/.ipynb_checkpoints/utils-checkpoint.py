import json
import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources
import os
from warnings import warn
from time import time

from aptinf.share import _cached_configs


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def use_xenon_plot_style():
    """Set matplotlib plot style."""
    params = {
        'font.family': 'serif',
        'font.size': 24, 'axes.titlesize': 24,
        'axes.labelsize': 24, 'axes.linewidth': 2,
        # ticks
        'xtick.labelsize': 22, 'ytick.labelsize': 22, 'xtick.major.size': 16, 'xtick.minor.size': 8,
        'ytick.major.size': 16, 'ytick.minor.size': 8, 'xtick.major.width': 2, 'xtick.minor.width': 2,
        'ytick.major.width': 2, 'ytick.minor.width': 2, 'xtick.direction': 'in', 'ytick.direction': 'in',
        # markers
        'lines.markersize': 12, 'lines.markeredgewidth': 3, 'errorbar.capsize': 8, 'lines.linewidth': 3,
        'savefig.bbox': 'tight', 'legend.fontsize': 24,
        'backend': 'Agg', 'mathtext.fontset': 'dejavuserif', 'legend.frameon': False,
        # figure
        'figure.facecolor': 'w',
        'figure.figsize': (12, 8),
        # pad
        'axes.labelpad': 12,
        # ticks
        'xtick.major.pad': 6, 'xtick.minor.pad': 6,
        'ytick.major.pad': 3.5, 'ytick.minor.pad': 3.5,
        # colormap
    }
    plt.rcParams.update(params)


@export
def load_data(file_name: str):
    """Load data from file. The suffix can be ".csv", ".pkl"."""
    file_name = get_file_path(file_name)
    fmt = file_name.split('.')[-1]
    if fmt == 'csv':
        data = pd.read_csv(file_name)
    elif fmt == 'pkl':
        data = pd.read_pickle(file_name)
    else:
        raise ValueError(f'unsupported file format {fmt}!')
    return data


@export
def load_json(file_name: str):
    """Load data from json file."""
    with open(get_file_path(file_name), 'r') as file:
        data = json.load(file)
    return data


def add_extensions(module1, module2, base, force=False):
    """
    Add subclasses of module2 to module1

    When ComponentSim compiles the dependency tree,
    it will search in the appletree.plugins module for Plugin(as attributes).
    When building Likelihood, it will also search for corresponding Component(s)
    specified in the instructions(e.g. NRBand).

    So we need to assign the attributes before compilation.
    These plugins are mostly user defined.
    """
    # Assign the module2 as attribute of module1
    is_exists = module2.__name__ in dir(module1)
    if is_exists and not force:
        raise ValueError(
            f'{module2.__name__} already existed in {module1.__name__}, '
            f'do not re-register a module with same name',
        )
    else:
        if is_exists:
            print(f'You have forcibly registered {module2.__name__} to {module1.__name__}')
        setattr(module1, module2.__name__.split('.')[-1], module2)

    # Iterate the module2 and assign the single Plugin(s) as attribute(s)
    for x in dir(module2):
        x = getattr(module2, x)
        if not isinstance(x, type(type)):
            continue
        add_extension(module1, x, base, force=force)


def add_extension(module, subclass, base, force=False):
    """
    Add subclass to module
    Skip the class when it is base class.

    It is no allowed to assign a class which has same name to an already assigned class.
    We do not allowed class name covering!
    Please change the name of your class when Error shows itself.
    """
    if getattr(subclass, '_' + subclass.__name__ + '__is_base', False):
        return

    if issubclass(subclass, base) and subclass != base:
        is_exists = subclass.__name__ in dir(module)
        if is_exists and not force:
            raise ValueError(
                f'{subclass.__name__} already existed in {module.__name__}, '
                f'do not re-register a {base.__name__} with same name',
            )
        else:
            if is_exists:
                print(f'You have forcibly registered {subclass.__name__} to {module.__name__}')
            setattr(module, subclass.__name__, subclass)


@export
def _get_abspath(file_name):
    """Get the abspath of the file. Raise FileNotFoundError when not found in any subfolder"""
    for sub_dir in ('../test/templates',):
        p = os.path.join(_package_path(sub_dir), file_name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {file_name}')


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder"""
    return pkg_resources.resource_filename('appletree', f'{sub_directory}')


@export
def get_file_path(fname):
    """Find the full path to the resource file
    Try 5 methods in the following order

    #. fname begin with '/', return absolute path
    #. url_base begin with '/', return url_base + name
    #. can get file from _get_abspath, return appletree internal file path
    #. can be found in local installed ntauxfiles, return ntauxfiles absolute path
    #. can be downloaded from MongoDB, download and return cached path
    """
    # 1. From absolute path
    # Usually Config.default is a absolute path
    if fname.startswith('/'):
        return fname

    # 2. From local folder
    # Use url_base as prefix
    if 'url_base' in _cached_configs.keys():
        url_base = _cached_configs['url_base']

        if url_base.startswith('/'):
            fpath = os.path.join(url_base, fname)
            if os.path.exists(fpath):
                warn(f'Load {fname} successfully from {fpath}')
                return fpath

    # 3. From internal files
    try:
        return _get_abspath(fname)
    except FileNotFoundError:
        pass

    # 4. From local installed ntauxfiles
    if NT_AUX_INSTALLED:
        # You might want to use this, for example if you are a developer
        if fname in ntauxfiles.list_private_files():
            fpath = ntauxfiles.get_abspath(fname)
            warn(f'Load {fname} successfully from {fpath}')
            return fpath

    # 5. From MongoDB
    try:
        import straxen
        # https://straxen.readthedocs.io/en/latest/config_storage.html
        # downloading-xenonnt-files-from-the-database  # noqa

        # we need to add the straxen.MongoDownloader() in this
        # try: except NameError: logic because the NameError
        # gets raised if we don't have access to utilix.
        downloader = straxen.MongoDownloader()
        # FileNotFoundError, ValueErrors can be raised if we
        # cannot load the requested config
        fpath = downloader.download_single(fname)
        warn(f'Loading {fname} from mongo downloader to {fpath}')
        return fname  # Keep the name and let get_resource do its thing
    except (FileNotFoundError, ValueError, NameError, AttributeError):
        warn(f'Mongo downloader not possible or does not have {fname}')

    # raise error when can not find corresponding file
    raise RuntimeError(f'Can not find {fname}, please check your file system')


@export
def timeit(indent=""):
    """Use timeit as a decorator.
    It will print out the running time of the decorated function.
    """
    def _timeit(func, indent):
        name = func.__name__

        def _func(*args, **kwargs):
            print(indent + f' Function <{name}> starts.')
            start = time()
            res = func(*args, **kwargs)
            time_ = (time() - start) * 1e3
            print(indent + f' Function <{name}> ends! Time cost = {time_:.2f} msec.')
            return res

        return _func
    if isinstance(indent, str):
        return lambda func: _timeit(func, indent)
    else:
        return _timeit(indent, "")
