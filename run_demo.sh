#!/bin/bash
# Demo script for 3D Mesh Refinement Pipeline

# Ensure we are executing relative to the script directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Define variables for the run (modify these as needed)
DATA_DIR="./test_data"
MESH_NAME="concept_mesh" 

echo "=========================================="
echo "Starting 3D Mesh Refinement Pipeline Demo"
echo "Data Directory : $DATA_DIR"
echo "Mesh Name      : $MESH_NAME"
echo "=========================================="

# Run the pipeline with basic arguments
python run_pipeline.py \
    --root_dir "$DATA_DIR" \
    --mesh_name "$MESH_NAME" \
    --re_remesh \

echo "Pipeline execution completed!"
