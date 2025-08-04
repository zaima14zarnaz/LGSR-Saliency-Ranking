#!/bin/bash

# Navigate to the parent directory
cd "$(dirname "$0")/.."

# Run the Python script from the results directory
python3 results/metrics.py
