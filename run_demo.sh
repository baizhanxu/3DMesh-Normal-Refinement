#!/bin/bash
# Demo script for 3D Mesh Refinement Pipeline

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MESH_PATH="${1:-./test_data/vase/647ba5cbafc1b3e480dbf73d83f51743.glb}"
CATEGORY="${2:-bottle}"

echo "=========================================="
echo "Starting 3D Mesh Refinement Pipeline Demo"
echo "Mesh Path : $MESH_PATH"
echo "Category  : $CATEGORY"
echo "=========================================="

python run_pipeline.py \
    --mesh_path "$MESH_PATH" \
    --cat "$CATEGORY"

echo "Pipeline execution completed!"
