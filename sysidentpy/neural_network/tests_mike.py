from narx_nn import *
import random

#test_object._basis_function_n_step_prediction
def main():
    test_object = NARXNN()
    
    test_object.model_type = "NARMAX"
    test_object.max_lag = 10
    
    try: test_object._basis_function_n_steps_horizon(1, 1, 1, 1)
    except: pass
    try: test_object._basis_function_n_steps_horizon([], [], 0, 0)
    except: pass
    try: test_object._basis_function_n_steps_horizon([random.randint(1, 100) for _ in range(1000)],[random.randint(1, 1000) for _ in range(1000)], 100, 100000)
    except: pass
    print(FLAG)


if __name__ == "__main__": main()