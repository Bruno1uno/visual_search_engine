import pytest
from pathlib import Path
from dashboard import load_metrics_json, check_api_health


def test_load_metrics_json_non_existent(tmp_path):
    """Test load_metrics_json returns empty dict for missing files."""
    missing_file = tmp_path / "missing.json"
    result = load_metrics_json(str(missing_file))
    assert result == {}


def test_load_metrics_json_valid(tmp_path):
    """Test load_metrics_json parses valid JSON data."""
    valid_file = tmp_path / "test_metrics.json"
    valid_file.write_text('{"loss_name": "proxy_anchor", "unseen_test_metrics": {"recall_at_1": 0.5243}}')

    result = load_metrics_json(str(valid_file))
    assert result["loss_name"] == "proxy_anchor"
    assert result["unseen_test_metrics"]["recall_at_1"] == 0.5243


def test_check_api_health_offline():
    """Test check_api_health handles offline server gracefully."""
    is_ready, count = check_api_health()
    # Should return False without throwing exceptions
    assert isinstance(is_ready, bool)
    assert isinstance(count, int)
