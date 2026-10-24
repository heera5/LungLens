"""
LungLens - Application Launcher
Run this script to start the web server from the project root.
"""
import os
import sys

# Ensure FRONTEND directory is in the import search path
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, 'FRONTEND')
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

# Change current working directory to FRONTEND so relative paths work seamlessly
os.chdir(frontend_dir)

from app import app, init_db

if __name__ == '__main__':
    print("=" * 60)
    print("  LungLens - Explainable Lung Cancer Screening System")
    print("  Local Server: http://127.0.0.1:5000")
    print("  Default Demo Clinician Login:")
    print("    Username: doctor")
    print("    Password: doctor123")
    print("=" * 60)
    init_db()
    # use_reloader=False: run.py changes the working directory, so Flask's auto-reloader
    # cannot relaunch it (that caused "can't open file ...\FRONTEND\run.py").
    app.run(debug=True, port=5000, use_reloader=False)
