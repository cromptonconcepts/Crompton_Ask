"""
TTM Ask - Launcher Service
A tiny HTTP server on port 5001 that starts Ollama and the main backend.
Started by Start TTM Ask.vbs and called by the HTML frontend.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
APP_PY = os.path.join(BASE_DIR, 'app.py')
DISCOVER_PY = os.path.join(BASE_DIR, 'discover_online_docs.py')
OLLAMA_KEEP_ALIVE = os.getenv('OLLAMA_KEEP_ALIVE', '-1')

# Windows flag: don't open a console window for spawned processes
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)

# Track background discovery process to avoid multiple parallel runs.
_discovery_proc = None


def _check_url(url):
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except Exception:
        return False


def is_ollama_running():
    return _check_url('http://localhost:11434/')


def is_backend_running():
    return _check_url('http://localhost:5000/health')


def start_services():
    global _discovery_proc
    started = []
    # Start Ollama if not already running
    if not is_ollama_running():
        try:
            ollama_env = os.environ.copy()
            ollama_env['OLLAMA_KEEP_ALIVE'] = OLLAMA_KEEP_ALIVE
            subprocess.Popen(
                ['ollama', 'serve'],
                creationflags=NO_WINDOW,
                cwd=BASE_DIR,
                env=ollama_env
            )
            started.append('ollama')
        except FileNotFoundError:
            started.append('ollama_not_found')

    # Start main Flask backend if not already running
    if not is_backend_running():
        subprocess.Popen(
            [PYTHON_EXE, APP_PY],
            creationflags=NO_WINDOW,
            cwd=BASE_DIR
        )
        started.append('backend')

    # Always trigger online discovery at startup (non-blocking).
    # Run once at a time to prevent overlapping discovery jobs.
    if os.path.exists(DISCOVER_PY):
        if _discovery_proc is None or _discovery_proc.poll() is not None:
            _discovery_proc = subprocess.Popen(
                [
                    PYTHON_EXE,
                    DISCOVER_PY,
                    '--download',
                    '--min-score', '2',
                    '--max-download', '12',
                    '--report', os.path.join(BASE_DIR, 'online_discovery_report.json')
                ],
                creationflags=NO_WINDOW,
                cwd=BASE_DIR
            )
            started.append('online_discovery')
        else:
            started.append('online_discovery_running')
    else:
        started.append('online_discovery_script_missing')

    return started


class LauncherHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress noisy access logs

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self._send_json({
                'launcher': True,
                'ollama': is_ollama_running(),
                'backend': is_backend_running()
            })
        elif self.path == '/start':
            started = start_services()
            self._send_json({'started': started})
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path == '/start':
            started = start_services()
            self._send_json({'started': started})
        else:
            self._send_json({'error': 'Not found'}, 404)


if __name__ == '__main__':
    server = HTTPServer(('localhost', 5001), LauncherHandler)
    print('TTM Ask launcher ready on http://localhost:5001')
    server.serve_forever()
