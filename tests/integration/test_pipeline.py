import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

def test_data_loader():
    """Test data loader component"""
    from src.data.data_loader import DataLoader
    
    loader = DataLoader()
    try:
        (x_train, y_train), (x_test, y_test) = loader.load_data()
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        print("DataLoader test passed")
        return True
    except Exception as e:
        print(f" DataLoader test failed: {e}")
        return False

def test_model_builder():
    """Test model builder component"""
    from src.models.model_builder import ModelBuilder
    
    builder = ModelBuilder()
    model = builder.create_cnn_model()
    builder.compile_model()
    
    assert model is not None
    assert len(model.layers) > 0
    print(" ModelBuilder test passed")
    return True

def test_config():
    """Test configuration setup"""
    from src.utils.config import config
    
    assert config.BASE_DIR.exists()
    assert config.DATA_DIR.exists()
    print("Config test passed")
    return True

def main():
    """Run all integration tests"""
    print("🧪 Running MLOps Pipeline Integration Tests...")
    
    tests = [test_data_loader, test_model_builder, test_config]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print(" All integration tests passed!")
        return True
    else:
        print(" Some tests failed")
        return False

if __name__ == "__main__":
    main()