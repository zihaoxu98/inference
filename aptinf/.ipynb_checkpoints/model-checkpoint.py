import numpy as np
import inspect
from scipy.special import loggamma


class ModelBase():
    arg_needed = set()
    obs_needed = set()

    def __init__(self, remap):
        self.remap = {}
        self.arg_remap = {}
        self.obs_remap = {}
        self.input_needed = set()
        self.arg_input_needed = set()
        self.obs_input_needed = set()
        self.set_remap(remap)

    def _set_input_needed_from_alias(self, alias, which):
        if type(alias) == str:
            self.input_needed.add(alias)
            if which == 'arg':
                self.arg_input_needed.add(alias)
            elif which == 'obs':
                self.obs_input_needed.add(alias)
        elif callable(alias):
            self.input_needed.update(inspect.getfullargspec(alias).args)
            if which == 'arg':
                self.arg_input_needed.update(inspect.getfullargspec(alias).args)
            elif which == 'obs':
                self.obs_input_needed.update(inspect.getfullargspec(alias).args)
        elif type(alias) == tuple:
            for _alias in alias:
                self._set_input_needed_from_alias(_alias, which)

    def set_remap(self, remap):
        """
        remap = {
            'arg': {arg_name: alias, ...},
            'obs': {obs_name: alias, ...},
        },
        alias can be a value, str (param name), a simple lambda func
        """
        assert self.arg_needed.issuperset(remap['arg'].keys()), "remap['arg'] must contain all args needed!"

        for arg_name, alias in remap['arg'].items():
            if arg_name in self.arg_needed:
                self.remap[arg_name] = alias
                self.arg_remap[arg_name] = alias
                self._set_input_needed_from_alias(alias, which='arg')
        for obs_name, alias in remap['obs'].items():
            if obs_name in self.obs_needed:
                self.remap[obs_name] = alias
                self.obs_remap[obs_name] = alias
                self._set_input_needed_from_alias(alias, which='obs')

    def _evaluate_item(self, inputs, item):
        value = item
        if type(value) == str:
            value = inputs[value]
        elif type(value) == tuple:
            value = tuple([self._evaluate_item(inputs, x) for x in value])
        elif callable(value):
            input_param_names = inspect.getfullargspec(value).args
            value = value(**{x: inputs[x] for x in input_param_names})
        return value

    def _get_values_from_remap(self, inputs, names_needed):
        result = {}
        for name in names_needed:
            result[name] = self._evaluate_item(inputs, self.remap[name])
        return result

    def likelihood(self, arg, obs):
        raise NotImplementedError

    def loglikelihood(self, arg, obs):
        raise NotImplementedError

    def simulate(self, arg):
        raise NotImplementedError

    def get_arg_from_inputs(self, inputs):
        return self._get_values_from_remap(inputs, self.arg_needed)

    def get_obs_from_inputs(self, inputs):
        return self._get_values_from_remap(inputs, self.obs_needed)

    def likelihood_from_inputs(self, inputs):
        return self.likelihood(self.get_arg_from_inputs(inputs),
                               self.get_obs_from_inputs(inputs))

    def loglikelihood_from_inputs(self, inputs):
        return self.loglikelihood(self.get_arg_from_inputs(inputs),
                                  self.get_obs_from_inputs(inputs))

    def simulate_from_inputs(self, inputs, **kwargs):
        return self.simulate(self.get_arg_from_inputs(inputs), **kwargs)

    def _get_quick_view(self, alias):
        if callable(alias):
            return inspect.getsource(alias).split(': ')[-1].replace('\n', '').replace('self.', '')
        elif type(alias) == tuple:
            return tuple([self._get_quick_view(x) for x in alias])
        else:
            return alias.__str__()

    def view(self):
        view = f"{self.__class__.__name__}(\n"
        for obs_name, alias in self.obs_remap.items():
            alias_view = self._get_quick_view(alias) if alias != 'data' else 'data'
            view += f"\t{obs_name} = {alias_view}\n"
        for arg_name, alias in self.arg_remap.items():
            alias_view = self._get_quick_view(alias) if alias != 'data' else 'data'
            view += f"\t{arg_name} = {alias_view}\n"
        view += ")"
        print(view)


class Gaussian(ModelBase):
    arg_needed = {'mu', 'std'}
    obs_needed = {'x'}

    def loglikelihood(self, arg, obs):
        return -0.5 * np.log(2 * np.pi * arg['std']**2) - 0.5 * (obs['x'] - arg['mu']**2) / arg['std']**2

    def simulate(self, arg, size=None):
        return np.random.normal(arg['mu'], arg['std'], size=size)


class Poisson(ModelBase):
    arg_needed = {'lam'}
    obs_needed = {'n'}

    def loglikelihood(self, arg, obs):
        return obs['n'] * np.log(arg['lam']) - arg['lam'] - loggamma(obs['n'] + 1)

    def simulate(self, arg, size=None):
        return np.random.poisson(arg['lam'], size=size)

class SingleSourceUnbinnedPDF(ModelBase):
    arg_needed = {'template'}
    obs_needed = {'events'}

    def loglikelihood(self, arg, obs):
        return np.log(arg['template'].pdf(obs['events'])).sum()

    def simulate(self, arg, size=1):
        return arg['template'].hist.get_random(size=size)


class MultiSourceUnbinnedPDF(ModelBase):
    arg_needed = {'templates', 'weights'}
    obs_needed = {'events'}

    def loglikelihood(self, arg, obs):
        templates = arg['templates']
        probs = np.array(arg['weights']) / np.sum(arg['weights'])
        events = obs['events']

        likelihood = 0
        for template, prob in zip(templates, probs):
            likelihood += template.pdf(events) * prob
        return np.sum(np.log(likelihood))

    def simulate(self, arg, size=1):
        probs = np.array(arg['weights']) / np.sum(arg['weights'])
        num_events = np.random.multinomial(size, probs)
        events = []
        for n, template in zip(num_events, arg['templates']):
            events.append(template.hist.get_random(size=n))
        return np.concatenate(events)
