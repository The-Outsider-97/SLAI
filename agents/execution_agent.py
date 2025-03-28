import subprocess

class ExecutionAgent:
    def __init__(self):
        pass

    def run_shell_command(self, command: str) -> str:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        except Exception as e:
            return str(e)

    def call_api(self, url: str, method: str = "GET", data: dict = None) -> str:
        import requests
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            else:
                return "Unsupported method"
            return response.text
        except Exception as e:
            return str(e)
