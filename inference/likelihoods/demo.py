import numpy as np

from inference.config import takes_config, Template, Constant
from inference.likelihood import ProfiledLikelihood
from inference.model import Poisson, Gaussian, MultiSourceUnbinnedPDF


@takes_config(
    Template(name='er',
             default='ER/template_XENONnT_ER_fv_default.h5',
             hist_name='cs1-log10_cs2-r'),
    Template(name='nr',
             default='RG/template_XENONnT_RG_fv_default.h5',
             hist_name='cs1-log10_cs2-r'),
    Template(name='wimp',
             default='WIMP-50GeV/template_XENONnT_WIMP-50GeV_fv_default.h5',
             hist_name='cs1-log10_cs2-r'),
    Constant(name='nr_rate_relative_std',
             default=0.1),
)
class testLikelihood(ProfiledLikelihood):
    def __init__(self):
        super().__init__()

        # Here are some small function defintions
        nr_rate = lambda lg_nr_rate: 10**lg_nr_rate
        nr_rate_mu = lambda: self.nr.norm
        nr_rate_std = lambda: self.nr.norm * self.nr_rate_relative_std
        rate_total = lambda lg_nr_rate, lg_er_rate, lg_sig_mul: 10**lg_nr_rate + 10**lg_er_rate + self.wimp.norm * 10**lg_sig_mul
        sources = lambda: (self.nr, self.er, self.wimp)
        sources_weights = lambda lg_nr_rate, lg_er_rate, lg_sig_mul: (10**lg_nr_rate, 10**lg_er_rate, self.wimp.norm * 10**lg_sig_mul)
        num_data = lambda data: len(data)

        # Here defines the likelihood
        config = [
            {'tag': 'poiss_tot',
             'model': Poisson,
             'arg': {'lam': rate_total},
             'obs': {'n': num_data}},

            {'tag': 'unbinned_pdf',
             'model': MultiSourceUnbinnedPDF,
             'arg': {'templates': sources, 'weights': sources_weights},
             'obs': {'events': 'data'}},

            {'tag': 'anc_nr_rate',
             'model': Gaussian,
             'arg': {'mu': nr_rate_mu, 'std': nr_rate_std},
             'obs': {'x': nr_rate}},
        ]
        self.set_likelihood(config)

        # Here we set the allowed range of parameters
        param_range = {
            'lg_er_rate': (-50, 50),
            'lg_nr_rate': (-50, 50),
            'lg_sig_mul': (-50, 50),
        }
        self.set_param_range(param_range)

    def simulate(self, param_for_simulate):
        # This function is to define how the toyMC is done.
        # There's no unique and correct way, so users must manually define it.

        num_events = self.likelihood_entries['poiss_tot'].simulate_from_param(param_for_simulate)
        data = self.likelihood_entries['unbinned_pdf'].simulate_from_param(param_for_simulate, size=num_events)
        return {'data': data}
