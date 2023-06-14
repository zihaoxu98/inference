import numpy as np

from aptinf.config import takes_config, Template, Constant
from aptinf.likelihood import ProfiledLikelihood
from aptinf.model import Poisson, Gaussian, MultiSourceUnbinnedPDF


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

        nr_rate = lambda log_nr_rate: 10**log_nr_rate
        nr_rate_mu = lambda: self.nr.norm
        nr_rate_std = lambda: self.nr.norm * self.nr_rate_relative_std
        rate_total = lambda log_nr_rate, log_er_rate, log_sig_mul: 10**log_nr_rate + 10**log_er_rate + self.wimp.norm * 10**log_sig_mul
        sources = lambda: (self.nr, self.er, self.wimp)
        sources_weights = lambda log_nr_rate, log_er_rate, log_sig_mul: (10**log_nr_rate, 10**log_er_rate, self.wimp.norm * 10**log_sig_mul)
        num_data = lambda data: len(data)

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

    def simulate(self, parameters_for_simulate=None):
        if parameters_for_simulate is None:
            parameters_for_simulate = {
                'log_er_rate': 2.0,
                'log_nr_rate': np.log10(self.nr.norm),
                'log_sig_mul': -99,
            }
        num_events = self.likelihood_entries['poiss_tot'].simulate_from_inputs(parameters_for_simulate)
        data = self.likelihood_entries['unbinned_pdf'].simulate_from_inputs(parameters_for_simulate, size=num_events)
        return {'data': data}