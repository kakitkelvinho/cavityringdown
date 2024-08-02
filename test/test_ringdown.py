import unittest
import numpy as np
import sys

sys.path.append('../analysis')

from analysis.ringdown import Ringdown

class TestRingdown(unittest.TestCase):

    def test_fit(self):
        t = np.arange(0, 2e-6, 2.5e-10)
        a, tau, c = [0.4, 1.8e-6, 0.03]
        trace = a*np.exp(-t/tau) - c
        trace += np.random.normal(0, a/5, len(trace))

        ringdown = Ringdown(timetrace=trace, t=t)

        popt, pcov = ringdown.fit(plot=False)

        # test if the fitted parameters are almost wqual

        message_tau = f"decay time tau is not fitted properly,\
                         fit is {popt[1]} while tau is {tau}."
        self.assertAlmostEqual(tau, popt[1], 4, message_tau)




if __name__ == '__main__':
    unittest.main()
