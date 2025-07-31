#!/usr/bin/env python3
"""
A1111 Ollama Extension Installation Script

This script handles the installation and setup of the A1111 Ollama extension.
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required Python packages."""
    requirements = [
        "requests",
        "gradio",
        "sqlite3"
    ]
    
    for requirement in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"Successfully installed {requirement}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {requirement}: {e}")
            return False
    return True

def setup_database():
    """Initialize the local database."""
    # TODO: Implement database setup
    print("Setting up local database...")
    pass

def main():
    """Main installation function."""
    print("Installing A1111 Ollama Extension...")
    
    if install_dependencies():
        print("Dependencies installed successfully.")
    else:
        print("Failed to install dependencies.")
        return False
        
    setup_database()
    print("Installation completed successfully!")
    return True

if __name__ == "__main__":
    main()
