import numpy as np
from immutabledict import immutabledict

class Likelihood():
    takes_config = immutabledict()
    
    def __init__(self):
        for config in self.takes_config.values():
            config.build()
        for config in self.takes_config.values():
            setattr(self, config.name, deepcopy(config))
    
    def set_likelihood(self, config):
        self.likelihood_entries = {}
        self.input_needed = set()

        for item in config:
            tag = item['tag']
            model_class = item['model']
            remap = {'arg': item['arg'], 'obs': item['obs']}
            
            model = model_class(remap)
            self.likelihood_entries[tag] = model
            self.input_needed.update(model.input_needed)

    def view(self):
        for model in self.likelihood_entries:
            model.view()
    
    def loglikelihood(self, inputs):
        return np.sum([model.loglikelihood_from_inputs(inputs) for model in self.likelihood_entries.values()])
    
    def simulate(self):
        raise NotImplementedError


def combine_likelihoods(*likelihoods):
    """Some function that can combine several likelihood classes into one."""
    pass
