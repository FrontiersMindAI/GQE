#!/bin/bash

# Script to run tests for the GQE package

set -e

echo "Using Python:"
python3 --version

if [ "$1" == "lightning" ]; then
    echo "Running tests with lightning support..."
    if command -v hatch >/dev/null 2>&1; then
        hatch run test-lightning:lightning
    else
        python3 -m pytest tests/test_gqe_lightning_callback.py -v
    fi
elif [ "$1" == "all" ]; then
    echo "Running all tests (including lightning if available)..."
    if command -v hatch >/dev/null 2>&1; then
        hatch run test-lightning:all
    else
        python3 -m pytest tests/ -v
    fi
else
    echo "Running tests with default setup (excluding lightning)..."
    if command -v hatch >/dev/null 2>&1; then
        hatch run test:default
    else
        python3 -m pytest tests/ -v -k "not lightning"
    fi
fi

echo "All tests completed successfully!"
