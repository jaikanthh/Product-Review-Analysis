#!/usr/bin/env python3
"""
Dashboard Launcher Script

This script launches the Product Review Analytics Dashboard with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the environment for running the dashboard."""
    # Add src directory to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(src_path)
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/warehouse",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        "streamlit": "streamlit",
        "plotly": "plotly", 
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn"
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def run_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "src" / "analytics" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    print("🚀 Launching Product Review Analytics Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#1f77b4",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f0f2f6"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("🎯 Product Review Analytics Dashboard Launcher")
    print("=" * 60)
    
    # Setup environment
    print("🔧 Setting up environment...")
    setup_environment()
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies satisfied!")
    print()
    
    # Run dashboard
    success = run_dashboard()
    
    if success:
        print("✅ Dashboard session completed successfully!")
    else:
        print("❌ Dashboard failed to start properly")
        sys.exit(1)

if __name__ == "__main__":
    main()