#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Go up two levels to reach 'Code/' (which contains LGSR/)
cd ../..


# Run Python with Code/ as PYTHONPATH so LGSR is recognized
PYTHONPATH=$(pwd) python3 -m Code.model.run_model
