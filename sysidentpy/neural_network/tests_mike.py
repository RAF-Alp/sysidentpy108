import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from narx_nn import FLAG, NARXNN
import random
import sys

# test_object._basis_function_n_step_prediction
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
        self.model = NARXNN()
    

    @patch.object(NARXNN, '_basis_function_predict', return_value=np.random.rand(10))
    def test_basis_function_n_step_prediction(self, mock_predict):
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10
        result = self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))
        self.assertTrue(mock_predict.called)

    @patch.object(NARXNN, '_basis_function_predict', return_value=np.random.rand(10))
    def test_basis_function_n_steps_horizon(self, mock_predict):
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10
        result = self.model._basis_function_n_steps_horizon(X, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))
        self.assertTrue(mock_predict.called)

    def test_value_error_insufficient_initial_conditions(self):
        X = np.random.rand(10, 1)
        y = np.random.rand(2, 1)
        steps_ahead = 2
        forecast_horizon = 10
        with self.assertRaises(ValueError):
            self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
            
    def test_value_error_insufficient_initial_conditions2(self):
        X = None
        y = np.random.rand(2, 1)
        steps_ahead = 2
        forecast_horizon = 10
        with self.assertRaises(ValueError):
            self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
            
    def test_value_error_insufficient_initial_conditions3(self):
        X = np.random.rand(10, 1)
        y = [0,0]
        steps_ahead = 2
        forecast_horizon = 10
        with self.assertRaises(ValueError):
            temp = self.model.max_lag
            self.model.max_lag = 1000000
            self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
            self.model.max_lag = temp
            
            
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

    def test_nar_model_type(self):
        self.model = NARXNN(model_type="NAR")
        #self.model.model_type = "NAR"
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10
        result = self.model._basis_function_n_step_prediction(None, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))
        self.model.model_type = "NAR"
        result = self.model._basis_function_n_steps_horizon(None, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))
        

    def test_nfir_model_type(self):
        self.model.model_type = "NFIR"
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10
        result = self.model._basis_function_n_step_prediction(X, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))
        self.model = NARXNN(model_type="NFIR")
        self.model.model_type = "NFIR"
        X = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
        steps_ahead = 2
        forecast_horizon = 10
        result = self.model._basis_function_n_steps_horizon(X, y, steps_ahead, forecast_horizon)
        self.assertEqual(result.shape, (forecast_horizon, 1))



if __name__ == '__main__':
    unittest.main(exit=False)
    for x, y in FLAG.items():
        print(f"Coverage for {x}:")
        for index, z in enumerate(y):
            if z: print(f"   Branch {index}: Reached")
            else: print(f"   Branch {index}: Not Reached")
        print(f"Function Coverage: {round(len([x for x in y if x]) / len(y) * 100, 2)}%\n")