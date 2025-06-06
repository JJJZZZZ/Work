#!/bin/bash

# Script to help launch the Data ChatBot website
echo "Launching Data ChatBot Website..."

# Check if Python is installed for a simple HTTP server
if command -v python3 &>/dev/null; then
    echo "Starting Python HTTP server..."
    cd "$(dirname "$0")"
    python3 -m http.server 8000
    echo "Server running at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
elif command -v python &>/dev/null; then
    echo "Starting Python HTTP server..."
    cd "$(dirname "$0")"
    python -m SimpleHTTPServer 8000
    echo "Server running at http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
else
    echo "Python not found. Please open index.html directly in your browser."
    # Try to open the file directly in the default browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$(dirname "$0")/index.html"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "$(dirname "$0")/index.html"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        start "$(dirname "$0")/index.html"
    else
        echo "Unable to automatically open the file. Please open index.html manually."
    fi
fi 