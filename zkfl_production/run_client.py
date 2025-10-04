#!/usr/bin/env python3
"""
Entry point script for ZKFL Production Client

This script provides a convenient way to start the ZKFL production client
with command-line configuration options.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from zkfl_production.client import main

if __name__ == "__main__":
    main()
