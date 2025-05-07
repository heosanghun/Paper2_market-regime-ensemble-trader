import pytest
from ensemble_controller import EnsembleController

def test_ensemble_controller_init():
    controller = EnsembleController()
    assert controller is not None

def test_ensemble_controller_run():
    controller = EnsembleController()
    # 예시: run()이 예외 없이 실행되는지 확인
    try:
        controller.run()
    except Exception as e:
        pytest.fail(f"run()에서 예외 발생: {e}") 