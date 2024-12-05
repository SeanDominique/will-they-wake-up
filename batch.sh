#!/bin/bash

# Define the list of arguments
args=(12 24 48)

# Iterate through the list and run the Python script with each argument
for arg in "${args[@]}"; do
  echo "Running training.py with argument: $arg"
  python3 training.py "$arg"
done
