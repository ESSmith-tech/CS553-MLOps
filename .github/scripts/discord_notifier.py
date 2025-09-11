"""
Discord notification script for test results.
Usage: python discord_notifier.py test_results.json <repository_name>
"""

import json, sys, os, requests
from typing import Dict, Any

def load_test_results(filepath: str) -> Dict[str, Any]:
    """Load and parse test results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_failure_message(data: Dict[str, Any], repository: str) -> str:
    """Format the failure message with numbered failed tests."""
    summary = data.get('summary', {})
    total_tests = summary.get('total', 0)
    failed_tests = summary.get('failed', 0)
    passed_tests = summary.get('passed', 0)
    errored_tests = summary.get('error', 0)

    if passed_tests == 0 and failed_tests == 0:
        broken_files = []
        for dictionary in data.get("collectors"):
            if "longrepr" in dictionary.keys():
                broken_files.append(dictionary["nodeid"])
        
        message_lines = [
            f" files in {repository} errored out: "
        ]

        for i, file_name in enumerate(broken_files, 1):
            message_lines.append(f"{i}. {file_name}")

        message_lines.append("")
        message_lines.append("Aborted pushing changes to HuggingFace")

        return "\n".join(message_lines)
    
    if failed_tests == 0 and errored_tests == 0:
        return f"ALL {total_tests} tests succeeded for {repository}. Changes pushed to HuggingFace."
    
    # Get failed test names
    failed_test_names = [
        test.get('nodeid', 'unknown') 
        for test in data.get('tests', []) 
        if test.get('outcome') == 'failed'
    ]

    errored_test_names = [
        test.get('nodeid', 'unknown') 
        for test in data.get('tests', []) 
        if test.get('outcome') == 'error'
    ]
    
    message_lines = [
        f"{failed_tests} out of {total_tests} tests failed for {repository}: "
    ]
    
    # Add numbered failed tests
    for i, test_name in enumerate(failed_test_names, 1):
        message_lines.append(f"{i}. {test_name}")

    if errored_tests > 0:
        message_lines.append(f"{errored_tests} out of {total_tests} tests errored for {repository}: ")
        for i, test_name in enumerate(errored_test_names, 1):
            message_lines.append(f"{i}. {test_name}")
    
    message_lines.append("")  # Empty line
    message_lines.append("Aborted pushing changes to HuggingFace")
    
    return "\n".join(message_lines)

def send_discord_notification(message: str, webhook_url: str) -> bool:
    """Send message to Discord via webhook."""
    payload = {"content": message}
    
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error sending Discord notification: {e}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python discord_notifier.py <test_results.json> <repository_name>", file=sys.stderr)
        sys.exit(1)
    
    test_results_file = sys.argv[1]
    repository = sys.argv[2]
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load test results
        test_data = load_test_results(test_results_file)
        
        # Format message
        message = format_failure_message(test_data, repository)
        
        # Send to Discord
        if send_discord_notification(message, webhook_url):
            print("Discord notification sent successfully")
        else:
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Error: Test results file '{test_results_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing test results JSON: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
