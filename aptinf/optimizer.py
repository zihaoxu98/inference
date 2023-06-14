import numpy as np
import scipy
from iminuit import Minuit


def minimize(func, x0, method='sequence'):
    if method == 'minuit':
        minimizer = Minuit(func, x0)
        minimizer.errordef = Minuit.LIKELIHOOD
        minimizer.migrad()
        if not minimizer.valid:
            raise RuntimeError("Minuit minimizer fails!")
        x = np.array(minimizer.values)
        y = minimizer.fmin.fval
        return x, y

    elif method == 'scipy':
        minimizer = scipy.optimize.minimize(func, x0, method='Powell')
        if not minimizer.success:
            raise RuntimeError("Scipy minimizer fails!")
        x = minimizer.x
        y = minimizer.fun
        return x, y

    elif method == 'sequence':
        for method in ['minuit', 'scipy']:
            try:
                return minimize(func, x0, method)
            except:
                pass
        raise RuntimeError("All the minimizers fail!")


def maximize(func, x0):
    x, y = minimize(lambda x: -func(x), x0)
    return x, -y
