import numpy as np
from copy import deepcopy
from immutabledict import immutabledict

from inference.optimizer import maximize
from inference.utils import to_array

class Likelihood():
    takes_config = immutabledict()
    
    def __init__(self):
        for config in self.takes_config.values():
            config.build()
        for config in self.takes_config.values():
            setattr(self, config.name, deepcopy(config))

    def set_likelihood(self, config):
        self.likelihood_entries = {}
        self.param_needed = set()
        self.param_range = {}

        for item in config:
            tag = item['tag']
            model_class = item['model']
            remap = {'arg': item['arg'], 'obs': item['obs']}

            model = model_class(remap)
            self.likelihood_entries[tag] = model
            self.param_needed.update(model.param_needed)

    def set_param_range(self, param_range):
        for key, range in param_range.items():
            self.param_range[key] = range

    def _get_param_bound_aux_llh(self, param):
        # This is needed to bypass the boundary issue in minimization
        auxiliary = 0
        for key, value in param.items():
            if key in self.param_range:
                if value < self.param_range[key][0]:
                    auxiliary += -np.exp(self.param_range[key][0] - value)
                if value > self.param_range[key][1]:
                    auxiliary += -np.exp(value - self.param_range[key][1])
        return auxiliary

    def view(self):
        for model in self.likelihood_entries.values():
            model.view()

    def loglikelihood(self, param):
        llh = self._get_param_bound_aux_llh(param)
        llh += np.sum([model.loglikelihood_from_param(param) for model in self.likelihood_entries.values()])
        return llh

    def simulate(self, param):
        raise NotImplementedError


class ProfiledLikelihood(Likelihood):
    def __init__(self):
        super().__init__()

    def _make_param(self, *param_dicts):
        param = {}
        for x in param_dicts:
            param.update(x)
        return param

    def set_data_from_toymc(self, param):
        self.set_data(self.simulate(param))

    def set_data(self, data):
        def _loglikelihood(param):
            param = self._make_param(data, param)
            llh = self._get_param_bound_aux_llh(param)
            llh += np.sum([model.loglikelihood_from_param(param) for model in self.likelihood_entries.values()])
            return llh
        self.loglikelihood = _loglikelihood

    def set_max_loglikelihood(self, param_guess):
        self.max_loglikelihood, bestfit = self.profiled_loglikelihood({}, param_guess, True)
        return bestfit

    def profiled_loglikelihood(self, param, param_profiled_guess, return_bestfit = False):
        if param_profiled_guess == {}:
            max_llh, bestfit = self.loglikelihood(param), {}
        else:
            sort_map = np.argsort(to_array(param_profiled_guess.keys()))
            param_profiled = to_array(param_profiled_guess.keys())[sort_map]
            param_profiled_guess = to_array(param_profiled_guess.values())[sort_map]
            def f(param_profiled_array):
                x = deepcopy(param)
                x.update({key: val for key, val in zip(param_profiled, param_profiled_array)})
                return self.loglikelihood(x)
            x, max_llh = maximize(f, param_profiled_guess)
            bestfit = {key: val for key, val in zip(param_profiled, x)}

        if return_bestfit:
            return max_llh, bestfit
        else:
            return max_llh

    def chi2(self, param, param_profiled_guess, return_bestfit=False):
        if not return_bestfit:
            return 2 * (self.max_loglikelihood - self.profiled_loglikelihood(param, param_profiled_guess))
        else:
            y, bestfit = self.profiled_loglikelihood(param, param_profiled_guess, True)
            llr = 2 * (self.max_loglikelihood - y)
            return llr, bestfit


class CombinedProfiledLikelihood(ProfiledLikelihood):
    def __init__(self, *likelihoods):
        self.likelihoods = likelihoods

        self.param_needed = set()
        self.param_range = dict()

        for _likelihood in self.likelihoods:
            self.param_needed.update(_likelihood.param_needed)

            for key, _range in _likelihood.param_range.items():
                if key in self.param_range:
                    lower = max(self.param_range[key][0], _range[0])
                    upper = min(self.param_range[key][1], _range[1])
                    self.param_range[key] = (lower, upper)
                else:
                    self.param_range[key] = _range

    def set_data_from_toymc(self, param):
        data = self.simulate(param)
        for _likelihood in self.likelihoods:
            _likelihood.set_data(data)

    def loglikelihood(self, param):
        llh = self._get_param_bound_aux_llh(param)
        for _likelihood in self.likelihoods:
            llh += _likelihood.loglikelihood(param)
        return llh

    def simulate(self, param):
        data = dict()
        for _likelihood in self.likelihoods:
            _data = _likelihood.simulate(param)
            # if np.any(np.isin(_data.keys(), data.keys())):
            #     raise ValueError("Conflict of toyMC outputs from different likelihoods!")
            # else:
            data.update(_data)
        return data

    def view(self):
        for _likelihood in self.likelihoods:
            _likelihood.view()

def combine_profile_likelihoods(*likelihoods):
    """Some function that can combine several likelihood classes into one."""
    return CombinedProfiledLikelihood(*likelihoods)
