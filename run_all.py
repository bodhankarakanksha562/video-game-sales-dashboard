#!/usr/bin/env python3
"""
Multi-service runner for Video Game Sales project.
Starts both Streamlit dashboard and FastAPI server.
"""

import subprocess
import sys
import time
import threading
import signal
import os

def run_streamlit():
    """Run Streamlit dashboard"""
    print("Starting Streamlit dashboard...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit failed to start: {e}")
    except KeyboardInterrupt:
        print("Streamlit stopped")

def run_api():
    """Run FastAPI server"""
    print("Starting FastAPI server...")
    try:
        subprocess.run([
            sys.executable, "api.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"API server failed to start: {e}")
    except KeyboardInterrupt:
        print("API server stopped")

def main():
    print("🎮 Video Game Sales Multi-Service Runner")
    print("=" * 50)
    print("This will start both services:")
    print("• Streamlit Dashboard: http://localhost:8501")
    print("• FastAPI Server: http://localhost:8000")
    print("• Frontend: Open frontend.html in browser")
    print("=" * 50)

    # Check if required files exist
    required_files = ["app.py", "api.py", "vgsales.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)

    print("✅ All required files found")

    # Start services in separate threads
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    api_thread = threading.Thread(target=run_api, daemon=True)

    try:
        print("🚀 Starting services...")
        api_thread.start()
        time.sleep(2)  # Give API a head start
        streamlit_thread.start()

        # Wait for threads
        streamlit_thread.join()
        api_thread.join()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main()