#!/usr/bin/env python3
"""
Unified Dashboard Launcher for Video Game Sales Application
Opens the combined dashboard in the default web browser
"""

import webbrowser
import time
import os
import sys

def open_unified_dashboard():
    """Open the unified dashboard in the default browser"""

    dashboard_path = os.path.join(os.getcwd(), "unified_dashboard.html")

    if not os.path.exists(dashboard_path):
        print("❌ Error: unified_dashboard.html not found!")
        print("Make sure you're running this from the project directory.")
        sys.exit(1)

    # Convert to file:// URL
    dashboard_url = f"file://{dashboard_path}"

    print("🎮 Video Game Sales - Unified Dashboard")
    print("=" * 50)
    print("Opening unified dashboard in your default browser...")
    print(f"File: {dashboard_path}")
    print()
    print("This dashboard combines:")
    print("• 📊 Streamlit Dashboard (Port 8501)")
    print("• ⚡ FastAPI Backend (Port 8000)")
    print("• 🌐 JavaScript Frontend (Port 8000)")
    print("• 💚 System Health Monitor")
    print()
    print("Make sure both services are running:")
    print("• python -m streamlit run app.py")
    print("• python api.py")
    print("=" * 50)

    # Give user time to read the message
    time.sleep(2)

    # Open in browser
    try:
        webbrowser.open(dashboard_url)
        print("✅ Dashboard opened successfully!")
        print("If it doesn't open automatically, manually open:")
        print(f"file://{dashboard_path}")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print("Please manually open:")
        print(f"file://{dashboard_path}")

if __name__ == "__main__":
    open_unified_dashboard()