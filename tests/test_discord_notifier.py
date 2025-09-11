import json, os, tempfile, pytest, sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.github', 'scripts'))

from discord_notifier import (
    load_test_results,
    format_failure_message,
    send_discord_notification,
    main
)

class TestLoadTestResults:
    """Test cases for load_test_results function."""
    
    def test_load_valid_json(self):
        """Test loading a valid JSON file."""
        test_data = {"summary": {"total": 5, "failed": 2}, "tests": []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_filename = f.name
        
        try:
            result = load_test_results(temp_filename)
            assert result == test_data
        finally:
            os.unlink(temp_filename)
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_test_results("nonexistent_file.json")
    
    def test_load_invalid_json(self):
        """Test loading an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_filename = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_test_results(temp_filename)
        finally:
            os.unlink(temp_filename)


class TestFormatFailureMessage:
    """Test cases for format_failure_message function."""
    
    def test_all_tests_passed(self):
        """Test message formatting when all tests pass."""
        data = {
            "summary": {"total": 5, "failed": 0, "passed": 5},
            "tests": []
        }
        repository = "my-org/my-repo"
        
        result = format_failure_message(data, repository)
        expected = "ALL 5 tests succeeded for my-org/my-repo. Changes pushed to HuggingFace."
        
        assert result == expected
    
    def test_some_tests_failed(self):
        """Test message formatting when some tests fail."""
        data = {
            "summary": {"total": 5, "failed": 2, "passed": 3},
            "tests": [
                {"nodeid": "tests/test_smoke.py::test_i_fail_always", "outcome": "failed"},
                {"nodeid": "tests/test_smoke.py::test_i_also_fail", "outcome": "passed"},
                {"nodeid": "tests/test_other.py::test_another_failure", "outcome": "failed"}
            ]
        }
        repository = "my-org/my-repo"
        
        result = format_failure_message(data, repository)
        expected = """2 out of 5 tests failed for my-org/my-repo: 
1. tests/test_smoke.py::test_i_fail_always
2. tests/test_other.py::test_another_failure

Aborted pushing changes to HuggingFace"""
        
        assert result == expected
    
    def test_failed_test_without_nodeid(self):
        """Test handling of failed tests without nodeid."""
        data = {
            "summary": {"total": 3, "failed": 1, "passed": 2},
            "tests": [
                {"outcome": "failed"},  # Missing nodeid
                {"nodeid": "tests/test_good.py::test_pass", "outcome": "passed"}
            ]
        }
        repository = "test-repo"
        
        result = format_failure_message(data, repository)
        
        assert "1. unknown" in result
        assert "1 out of 3 tests failed" in result
    
    def test_missing_tests_key(self):
        """Test handling when tests key is missing."""
        data = {
            "summary": {"total": 2, "failed": 1, "passed": 1}
            # Missing "tests" key
        }
        repository = "test-repo"
        
        result = format_failure_message(data, repository)
        
        # Should handle gracefully with no failed tests listed
        assert "1 out of 2 tests failed" in result
        assert "Aborted pushing changes to HuggingFace" in result

class TestIntegration:
    """Integration tests that test multiple functions together."""
    
    def test_complete_failure_flow(self):
        """Test the complete flow for a test failure scenario."""
        # Create test data
        test_data = {
            "summary": {"total": 4, "failed": 2, "passed": 2},
            "tests": [
                {"nodeid": "tests/test_smoke.py::test_i_fail_always", "outcome": "failed"},
                {"nodeid": "tests/test_smoke.py::test_i_pass", "outcome": "passed"},
                {"nodeid": "tests/test_other.py::test_another_failure", "outcome": "failed"},
                {"nodeid": "tests/test_other.py::test_another_pass", "outcome": "passed"}
            ]
        }
        
        # Test the formatting
        result = format_failure_message(test_data, "my-org/test-repo")
        
        expected_lines = [
            "2 out of 4 tests failed for my-org/test-repo: ",
            "1. tests/test_smoke.py::test_i_fail_always",
            "2. tests/test_other.py::test_another_failure",
            "",
            "Aborted pushing changes to HuggingFace"
        ]
        
        assert result == "\n".join(expected_lines)
    
    @patch('discord_notifier.requests.post')
    def test_complete_success_flow(self, mock_post):
        """Test the complete flow for a success scenario."""
        # Setup successful HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create test data for all passing tests
        test_data = {
            "summary": {"total": 3, "failed": 0, "passed": 3},
            "tests": [
                {"nodeid": "tests/test_one.py::test_pass1", "outcome": "passed"},
                {"nodeid": "tests/test_one.py::test_pass2", "outcome": "passed"},
                {"nodeid": "tests/test_two.py::test_pass3", "outcome": "passed"}
            ]
        }
        
        # Format message
        message = format_failure_message(test_data, "test-repo")
        
        # Send notification
        result = send_discord_notification(message, "https://discord.com/webhook")
        
        assert result is True
        assert "ALL 3 tests succeeded" in message
        mock_post.assert_called_once_with(
            "https://discord.com/webhook",
            json={"content": message},
            headers={"Content-Type": "application/json"}
        )
