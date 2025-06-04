#!/usr/bin/env python3
"""
Startup script for the Propensity Matching Tool Web Application
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Start the web application"""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print("🚀 Starting Propensity Matching Tool Web Application...")
    print(f"📁 Working directory: {script_dir}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📖 Check the README.md for usage instructions")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start the FastAPI application
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
