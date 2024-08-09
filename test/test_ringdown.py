import unittest
import numpy as np
import sys

sys.path.append('../analysis')

from analysis.ringdown import Ringdown

class TestRingdown(unittest.TestCase):

    def test_fit(self):
        '''Test to see whether we can recover the correct time decay constant'''
        a, tau, c = [0.4, 1.8e-6, 0.03]
        t, trace = self.generate_timetrace(a, tau, c, a/100 )

        ringdown = Ringdown(timetrace=trace, t=t)

        popt, _ = ringdown.fit(plot=False)

        # test if the fitted parameters are almost wqual

        message_tau = f"decay time tau is not fitted properly,\
                         fit is {popt[1]} while tau is {tau}."
        self.assertAlmostEqual(tau, popt[1], 4, message_tau)

    def test_fit2(self):
        '''Same test as above but with different parameters'''
        a, tau, c = [0.8, 1.2e-6, 0.]
        t, trace = self.generate_timetrace(a, tau, c, a/80)

        ringdown = Ringdown(timetrace=trace, t=t)
        popt, _ = ringdown.fit(plot=False)

        # test if the fitted parameters are almost wqual

        message_tau = f"decay time tau is not fitted properly,\
                         fit is {popt[1]} while tau is {tau}."
        self.assertAlmostEqual(tau, popt[1], 4, message_tau)


    def test_noise(self):
        '''Test to see whether the correct noice statistic can be recovered'''
        a, tau, c = [0.8, 1.2e-6, 0.]
        t, trace = self.generate_timetrace(a, tau, c, a/80)

        ringdown = Ringdown(timetrace=trace, t=t)

        std = ringdown.exp_residual()

        self.assertAlmostEqual(round(std*np.max(ringdown.timetrace),2), round(a/80,2), 2, "Noise statistic does not fit.")
        
        

    def generate_timetrace(self, a, tau, c, noise_sd, tEnd=2e-6, tInc=2.5e-10):
        '''Generates a timetrace of an exponential with a, tau, c
        and sprinkle in a normally distributed noise with a noise_sd.'''
        t = np.arange(0, tEnd, tInc)
        trace = a*np.exp(-t/tau) + c
        trace += np.random.normal(0, noise_sd, len(trace))

        return t, trace


if __name__ == '__main__':
    unittest.main()
