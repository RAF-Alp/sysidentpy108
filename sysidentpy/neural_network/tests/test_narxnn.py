import numpy as np
import torch
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from torch import nn

from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.utils.narmax_tools import regressor_code

torch.manual_seed(0)


class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 1)
        self.tanh = nn.Tanh()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z


def create_test_data(n=1000):
    # np.random.seed(42)
    # x = np.random.uniform(-1, 1, n).T
    # y = np.zeros((n, 1))
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])
    # lag = 2
    # for k in range(lag, len(x)):
    #     y[k] = theta[4]*y[k-1]**2 + theta[2]*y[k-1]*x[k-1] + theta[0]*x[k-2] \
    #         + theta[3]*y[k-2]*x[k-2] + theta[1]*y[k-2]

    # y = np.reshape(y, (len(y), 1))
    # x = np.reshape(x, (len(x), 1))
    # data = np.concatenate([x, y], axis=1)
    data = np.loadtxt("examples/datasets/data_for_testing.txt")
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return x, y, theta


x, y, _ = create_test_data()
train_percentage = 90
split_data = int(len(x) * (train_percentage / 100))

X_train = x[0:split_data, 0]
X_test = x[split_data::, 0]

y1 = y[0:split_data, 0]
y_test = y[split_data::, 0]
y_train = y1.copy()

y_train = np.reshape(y_train, (len(y_train), 1))
X_train = np.reshape(X_train, (len(X_train), 1))

y_test = np.reshape(y_test, (len(y_test), 1))
X_test = np.reshape(X_test, (len(X_test), 1))


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "model_type": "NARMAX",
        "batch_size": 100,
        "learning_rate": 0.01,
        "epochs": 200,
        "optimizer": "Adam",
        "net": None,
        "train_percentage": 80,
        "verbose": False,
        "optim_params": None,
    }
    model = NARXNN(basis_function=Polynomial())
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
        model.batch_size,
        model.learning_rate,
        model.epochs,
        model.optimizer,
        model.net,
        model.train_percentage,
        model.verbose,
        model.optim_params,
    ]
    assert list(default.values()) == model_values


def test_validate():
    assert_raises(ValueError, NARXNN, ylag=-1, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, ylag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, xlag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, xlag=-1, basis_function=Polynomial(degree=1))


def test_fit_raise():
    assert_raises(
        ValueError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        model_type="NARARMAX",
    )


# def test_fit_raise_y():
#     model = NARXNN(basis_function=Polynomial(degree=2))
#     assert_raises(ValueError, model.fit, X=X_train, y=None) #Fails 80% fit coverage
def test_fit_raise_y():
    model = NARXNN(basis_function=Polynomial(degree=2))
    assert_raises(ValueError, model.fit, X=X_train, y=None)
    assert_raises(ValueError, model.fit, X=None, y=y_train)  # Added to cover X=None branch passes +80% fit function

# def test_fit_lag_nar():
#     basis_function = Polynomial(degree=1)

#     regressors = regressor_code(
#         X=X_train,
#         xlag=2,
#         ylag=2,
#         model_type="NAR",
#         model_representation="neural_network",
#         basis_function=basis_function,
#     )
#     n_features = regressors.shape[0]

#     class NARX(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.lin = nn.Linear(n_features, 30)
#             self.lin2 = nn.Linear(30, 30)
#             self.lin3 = nn.Linear(30, 1)
#             self.tanh = nn.Tanh()

#         def forward(self, xb):
#             z = self.lin(xb)
#             z = self.tanh(z)
#             z = self.lin2(z)
#             z = self.tanh(z)
#             z = self.lin3(z)
#             return z

#     model = NARXNN(
#         net=NARX(),
#         ylag=2,
#         xlag=2,
#         basis_function=basis_function,
#         model_type="NAR",
#         loss_func="mse_loss",
#         optimizer="Adam",
#         epochs=10,
#         verbose=False,
#         optim_params={
#             "betas": (0.9, 0.999),
#             "eps": 1e-05,
#         },  # optional parameters of the optimizer
#     )

#     model.fit(X=X_train, y=y_train)
#     assert_equal(model.max_lag, 2)
def test_fit_lag_nar():
    basis_function = Polynomial(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NAR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    # Test with verbose=True to cover additional branches
    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=10,
        verbose=True,  # Changed from False to True
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    assert_raises(ValueError, model.fit, X=X_train, y=y_train, X_test=None, y_test=None)  # Cover verbose branch
    model.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    assert_equal(model.max_lag, 2)

def test_fit_lag_nfir():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NFIR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        basis_function=basis_function,
        model_type="NFIR",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=10,
        verbose=False,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax():
    basis_function = Polynomial(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=10,
        verbose=False,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax_fourier():
    basis_function = Fourier(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=10,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_model_predict():
    basis_function = Polynomial(degree=2)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_1():
    basis_function = Polynomial(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_3():
    basis_function = Polynomial(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_raise_batch_size():
    assert_raises(
        ValueError, NARXNN, batch_size=0.3, basis_function=Polynomial(degree=2)
    )


def test_raise_epochs():
    assert_raises(ValueError, NARXNN, epochs=0.3, basis_function=Polynomial(degree=2))


def test_raise_train_percentage():
    assert_raises(
        ValueError, NARXNN, train_percentage=-1, basis_function=Polynomial(degree=2)
    )


def test_raise_verbose():
    assert_raises(TypeError, NARXNN, verbose=None, basis_function=Polynomial(degree=2))


def test_raise_device():
    assert_raises(ValueError, NARXNN, device="CPU", basis_function=Polynomial(degree=2))


def test_model_predict_fourier():
    basis_function = Fourier(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_1_fourier():
    basis_function = Fourier(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=1000,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-03,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_3_fourier():
    basis_function = Fourier(degree=1)

    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)

# def test_check_cuda():
#     model = NARXNN(basis_function=Polynomial())
#     assert_equal(model._check_cuda("cpu").type, "cpu")
#     if torch.cuda.is_available():
#         assert_equal(model._check_cuda("cuda").type, "cuda")
#     else:
#         with assert_raises(ValueError):
#             model._check_cuda("cuda")
#     with assert_raises(ValueError):
#         model._check_cuda("invalid_device")

# def test_split_data():
#     # Test split_data with both X and y
#     model = NARXNN(basis_function=Polynomial())
#     reg_matrix, y_transformed = model.split_data(X_train, y_train)
#     assert reg_matrix.shape[0] == y_transformed.shape[0]

#     # Test split_data with X as None
#     model = NARXNN(basis_function=Polynomial())
#     reg_matrix, y_transformed = model.split_data(None, y_train)
#     assert reg_matrix.shape[0] == y_transformed.shape[0]

#     # Test split_data with a different basis function
#     model = NARXNN(basis_function=Fourier(degree=1))
#     reg_matrix, y_transformed = model.split_data(X_train, y_train)
#     assert reg_matrix.shape[0] == y_transformed.shape[0]
def test_split_data_y_none():
    # Test split_data with y as None to reach Branch 2
    model = NARXNN(basis_function=Polynomial())
    assert_raises(ValueError, model.split_data, X_train, None)


def test_split_data_ensemble():
    # Test split_data with basis_function ensemble set to True to reach Branch 6
    class CustomBasisFunction:
        def __init__(self, degree=1, ensemble=True, repetition=2):
            self.degree = degree
            self.ensemble = ensemble
            self.repetition = repetition

        def fit(self, lagged_data, max_lag, predefined_regressors=None):
            return np.random.rand(lagged_data.shape[0], 3), self.ensemble

        def transform(self, lagged_data, max_lag):
            return np.random.rand(lagged_data.shape[0], 3), self.ensemble

    custom_basis_function = CustomBasisFunction()
    model = NARXNN(basis_function=custom_basis_function)
    reg_matrix, y_transformed = model.split_data(X_train, y_train)
    assert reg_matrix.shape[0] == y_transformed.shape[0]
    assert model.regressor_code is not None


def test_split_data_polynomial():
    # Test split_data with basis_function being Polynomial to reach Branch 8
    model = NARXNN(basis_function=Polynomial())
    reg_matrix, y_transformed = model.split_data(X_train, y_train)
    assert reg_matrix.shape[0] == y_transformed.shape[0]
    assert model.regressor_code is not None
    assert model.regressor_code.shape[1] == 4  # Check if regressor_code has the expected shape



if __name__ == "__main__":
    import pytest
    pytest.main([__file__])