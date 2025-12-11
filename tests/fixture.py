import numpy as np
import pytest

@pytest.fixture(scope = "function")
def rmm_data():
    return np.load("./tests/rmm.npy")