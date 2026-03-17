# 3D Mesh Refinement Pipeline

This repository contains a robust pipeline for enhancing and refining 3D meshes using AI-generated normal maps and multi-view optimization. By leveraging advanced generative models and explicit geometry optimization, it adds high-fidelity details to coarse or generated 3D models while preserving global structural integrity.

## Overview

The pipeline executes the following automated steps:

1. **Multi-view Normal Rendering**: Renders the initial coarse 3D mesh into multi-view normal maps (typically 4 views: front, back, left, right), merged into a single layout.
2. **AI-Driven Detail Generation**: Uses generative AI (e.g., Gemini) to add rich, photorealistic, and structurally coherent details to the multi-view normal maps based on the object's category.
3. **Background Removal & Processing**: Applies background removal and optional Super Resolution (SR) to the generated AI normal maps to ensure clean and high-quality reference signals.
4. **View Splitting**: Separates the merged layout back into distinct directional views.
5. **Geometry Optimization (`mv_refine.py`)**: 
   - Optimizes the 3D mesh geometry to align with the AI-generated normal maps via differentiable rendering.
   - Utilizes **Frequency Separation** to balance between injecting high-frequency micro-details and allowing macro-contour/silhouette deformations.

## Environment Setup
You can set up the environment using Conda. Run the following commands in your terminal:
```
conda create -n normal_refine python=3.10 -y
conda activate normal_refine

pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt --no-build-isolation --no-deps
```

## Directory Structure

- `run_pipeline.py`: The main controller script that orchestrates the entire workflow.
- `code/mv_refine.py`: The core PyTorch-based 3D multi-view geometry optimization engine.
- `code/render_init_normals_merge.py`: Script for rendering initial multi-view normals.
- `code/api_gemini_gen_img.py`: Interface for generating enhanced normal maps using the Gemini API.
- `code/rb_img.py`: Background removal utility.

## Usage

We provide a Bash script demo `run_demo.sh` for easy execution. You can also run the Python script directly using command-line arguments.

### 1. Directory Setup
Ensure your mesh files are organized in the following structure:
```data/
└── {root_dir}/
    └──  {category}_{id}/
         └── {mesh_name}.obj/glb
```
where:
- `{root_dir}`: Base directory for your mesh cases (e.g., `test_cases`).
- `{category}`: Object category (e.g., `chair`, `car`).
- `{id}`: Unique identifier for the mesh case. 
- `{mesh_name}`: Base name of the coarse 3D mesh files which you want to refine without extension.

For example:
```data/
└── test_cases/
    └── chair_001/
         └── our_method.obj
    └── bottle_001/
         └── our_method.obj 
```
### 2. Set Gemini API Key
Set the api_url and api_key in [code/api_gemini_gen_img.py](code/api_gemini_gen_img.py) to enable AI-driven normal map enhancement.

### 3. Using the Demo Script
You can run the entire pipeline with a single command using the provided demo script:
```bash
chmod +x run_demo.sh
./run_demo.sh
```
you can modify the `run_demo.sh` script to specify different root directories, mesh names.

### 4. Using Command Line
Or run the pipeline by providing the necessary arguments to `run_pipeline.py`:
```bash
python run_pipeline.py --root_dir "./data/test_cases" --mesh_name "our_method" --re_remesh
```

**Available Arguments in `run_pipeline.py`**:
- `--root_dir`: Directory containing your mesh cases (default: `./data/test_cases`).
- `--mesh_name`: Base name of the 3D mesh files without extension, e.g., `concept_mesh` (looks for `.obj` or `.glb`).
- `--re_gen`: Flag to force re-generation of AI images and background removal.
- `--re_remesh`: Flag to force re-execution of the 3D mesh refinement step (`mv_refine.py`).
- `--no_sr`: Flag to **disable** Super Resolution on the AI-generated normal maps.
- `--use_frequency_separation`: Flag to enable frequency separation. This blends geometry frequencies to construct high-frequency micro-details while preserving the overall clean global shape/macro-silhouette from the original coarse mesh.
- `--force_subdivide`: Flag to forcefully subdivide the initial mesh. Ensures the mesh reaches a sufficient vertex count (targets >10,000) for high-quality detail displacement if the initial grid is too sparse.

## Output
