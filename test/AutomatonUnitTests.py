from dataghobot.ghobot import Automaton
import unittest
import numpy as np


class OptimTestCase(unittest.TestCase):

    def test_enchance_params(self):
        params1 = {
            'ok': 1,
            'ko': 2
        }
        params2 = {
            'ok': 42,
            'hehe': 13
        }
        param_result = Automaton.enhance_param([params1, None], **params2)
        self.assertEqual(param_result[0]['ok'], 42)
        self.assertEqual(param_result[0]['ko'], 2)
        self.assertEqual(len(param_result[0]), 2)

    def test_stacking_res(self):
        a1 = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        a2 = np.array([[0.1, 0.9], [0.15, 0.85], [0.155, 0.845]])
        a3 = np.array([[0.25, 0.75], [0.255, 0.745], [0.2555, 0.7445]])
        aa1 = [a1, a2, a3]
        aa2 = [a1, a2, a3]
        aa3 = [a1, a2, a3]
        aaa = [aa1, aa2, aa3]
        result = Automaton.stacking_res_to_one_pred(aaa)
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()
