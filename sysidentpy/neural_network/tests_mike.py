from narx_nn import *
import random
import unittest
import numpy as np
from narx_nn import FLAG

#test_object._basis_function_n_step_prediction
# def main():
#     test_object = NARXNN()
    
#     test_object.model_type = "NARMAX"
#     test_object.max_lag = 10
    
#     try: test_object._basis_function_n_steps_horizon(1, 1, 1, 1)
#     except: pass
#     try: test_object._basis_function_n_steps_horizon([], [], 0, 0)
#     except: pass
#     try: test_object._basis_function_n_steps_horizon([random.randint(1, 100) for _ in range(1000)],[random.randint(1, 1000) for _ in range(1000)], 100, 100000)
#     except: pass
#     print(FLAG)


class TestBasisFunctionMethods(unittest.TestCase):
    def setUp(self):
        self.max_lag = 3
        self.model_type = "NARMAX"
        self.model = NARXNN()

    def test_basis_function_n_step_prediction(self):
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10

        result = self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))

    def test_basis_function_n_steps_horizon(self):
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10

        result = self.model._basis_function_n_steps_horizon(X, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))

    def test_value_error_insufficient_initial_conditions(self):
        X = np.random.rand(10, 1)
        y = np.random.rand(2, 1)  # Insufficient initial conditions
        steps_ahead = 2
        forecast_horizon = 10

        with self.assertRaises(ValueError):
            self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)

    def test_value_error_invalid_model_type(self):
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10
        self.model.model_type = "INVALID"
        with self.assertRaises(ValueError):
            self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
        with self.assertRaises(ValueError):
            self.model._basis_function_n_steps_horizon(X, y, steps_ahead, forecast_horizon)
            
    def test_return_coverage(self):
        print(FLAG)

if __name__ == '__main__':
    unittest.main()
    unittest.test_return_coverage()
#if __name__ == "__main__": main()