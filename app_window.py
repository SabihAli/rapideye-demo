#!/usr/bin/env python3
"""
RapidEye Desktop Application Entry Point (`app_window.py`)
Launches the FastAPI backend in a background daemon thread and opens a native
OS desktop GUI window (Chromium/Edge on Windows, WebKit on macOS/Linux)
pointing to the locally served React application.
"""

import os
import sys
import time
import threading
import uvicorn
import webview

# Ensure current directory or bundle root is in Python path
if hasattr(sys, '_MEIPASS'):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.main import app
from server.config import settings

def run_server():
    """Start the FastAPI backend engine."""
    print("[Desktop Engine] Starting local backend server on http://127.0.0.1:8001...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="warning",
        access_log=False
    )

def main():
    # 1. Spawn FastAPI server in a background daemon thread so it terminates when the GUI closes
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 2. Give server a brief moment to initialize DB and routes
    time.sleep(1.5)

    # 3. Create native desktop window
    window = webview.create_window(
        title="RapidEye Security Monitor (Local Desktop Edition)",
        url="http://127.0.0.1:8001",
        width=1600,
        height=950,
        min_size=(1200, 700),
        confirm_close=True
    )

    print("[Desktop GUI] Opening desktop application window...")
    # Start the webview GUI loop (blocking call on main thread)
    webview.start(debug=False)
    
    print("[Desktop GUI] Application window closed. Shutting down services cleanly...")

if __name__ == "__main__":
    main()
