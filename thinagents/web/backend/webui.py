import webbrowser
import uvicorn
import threading
import subprocess
import time
import sys
from pathlib import Path


class WebUI:
    """
    Web UI for ThinAgents.
    
    Auto-detects dev mode if frontend source is available,
    otherwise uses production build.
    
    Args:
        agent: ThinAgents Agent instance
        host: Server host (default: "127.0.0.1")
        port: Server port (default: 8000)
        dev_mode: Force dev/prod mode (default: None = auto-detect)
    """
    def __init__(self, agent, host="127.0.0.1", port=8000, dev_mode=None):
        self.agent = agent
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        self._frontend_process = None

    def run(self, host=None, port=None, open_browser=True, dev_mode=None):
        host = host or self.host
        port = port or self.port
        
        frontend_src = Path(__file__).parent.parent.parent / "frontend" / "src"
        
        if dev_mode is None and self.dev_mode is None:
            auto_dev = frontend_src.exists()
            if auto_dev:
                print(f"🔍 Auto-detected dev mode (found frontend source at {frontend_src})")
        else:
            auto_dev = dev_mode if dev_mode is not None else self.dev_mode
        
        if auto_dev and frontend_src.exists():
            self._run_dev(host, port, open_browser)
        else:
            self._run_prod(host, port, open_browser)

    def _run_dev(self, host, port, open_browser):
        frontend_port = 5173
        print(f"\n🚀 ThinAgents Web UI (DEV)\n📍 Backend: http://{host}:{port}\n📍 Frontend: http://{host}:{frontend_port}\n")
        
        self._start_backend(host, port)
        time.sleep(2)
        self._start_frontend(host, frontend_port)
        
        if open_browser:
            threading.Timer(3, lambda: webbrowser.open(f"http://{host}:{frontend_port}")).start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            sys.exit(0)

    def _run_prod(self, host, port, open_browser):
        from .server import create_app
        
        build_dir = Path(__file__).parent.parent / "ui" / "build"
        if not build_dir.exists():
            print("❌ UI build not found! Run: python scripts/build_ui.py")
            sys.exit(1)

        app = create_app(self.agent)
        print(f"\n🚀 ThinAgents Web UI\n📍 Server: http://{host}:{port}\n")
        
        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        
        uvicorn.run(app, host=host, port=port, log_level="warning")

    def _start_backend(self, host, port):
        from .server import create_app
        app = create_app(self.agent)
        
        def run():
            uvicorn.run(app, host=host, port=port, log_level="warning")
        
        threading.Thread(target=run, daemon=True).start()

    def _start_frontend(self, host, port):
        frontend_dir = Path(__file__).parent.parent.parent / "frontend"
        
        try:
            self._frontend_process = subprocess.Popen(
                ["pnpm", "run", "dev", "--host", host, "--port", str(port)],
                cwd=frontend_dir,
            )
        except FileNotFoundError:
            print("❌ pnpm not found. Install: npm install -g pnpm")
            sys.exit(1)

    def stop(self):
        if self._frontend_process:
            self._frontend_process.terminate()
            try:
                self._frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._frontend_process.kill()
        print("✅ Stopped")

