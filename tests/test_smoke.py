# tests/test_smoke.py

import pytest
from omegaconf import OmegaConf

def test_imports():
    # Test that all key modules can be imported without error.
    try:
        from data import make_dataset
        from models import model_architectures, train_lightweight, train_ensembles, evaluate_models
        from visualization import visualize
    except ImportError as e:
        pytest.fail(f"Failed to import a module: {e}")

def test_config_loading():
    # Test if the default config file can be loaded
    try:
        config = OmegaConf.load("configs/config.yml")
        assert "paths" in config
        assert "data" in config
        assert "training" in config
        assert "models" in config
    except FileNotFoundError:
        pytest.fail("Default config file 'configs/config.yml' not found.")
    except Exception as e:
        pytest.fail(f"Failed to load or parse config file: {e}")

def test_architecture_functions_exist():
    from models import model_architectures
    assert hasattr(model_architectures, 'build_base_model')
    assert hasattr(model_architectures, 'build_averaging_ensemble')
    assert hasattr(model_architectures, 'build_stacking_ensemble')

def test_data_functions_exist():
    from data import make_dataset
    assert hasattr(make_dataset, 'load_and_split_data')
    assert hasattr(make_dataset, 'create_dataset')
    assert hasattr(make_dataset, 'get_class_weights')