#!/bin/bash

# Define the list of arguments
args=(0 12 24 36 48 60)

# Iterate through the list and run the Python script with each argument
for arg in "${args[@]}"; do
  echo "Running training.py with argument: $arg"
  python3 training.py "$arg"
done
