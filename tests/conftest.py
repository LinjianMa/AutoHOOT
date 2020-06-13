"""
Configuration file. Must resides in tests root directory.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--backendopt",
                     action="store",
                     nargs='+',
                     default=["numpy", "jax", "ctf", "tensorflow"],
                     help="A list of backends.")


@pytest.fixture
def backendopt(request):
    return request.config.getoption("--backendopt")
