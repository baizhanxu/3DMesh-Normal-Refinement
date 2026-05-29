# 3D Mesh Refinement Pipeline

This repository contains a robust pipeline for enhancing and refining 3D meshes using AI-generated normal maps and multi-view optimization. By leveraging advanced generative models and explicit geometry optimization, it adds high-fidelity details to coarse or generated 3D models while preserving global structural integrity.

## Overview

The pipeline supports two primary modes of operation:

- **Parallel Multi-view Mode (Default)**: Renders, generates, and optimizes across 4 or 6 specific views simultaneously.
- **Autoregressive Mode**: Progressively rotates the object (e.g., across 12 views), rendering the current state, inpainting only newly visible regions using Gemini, and jointly optimizing the mesh geometry step-by-step for higher multi-view consistency.

Depending on the mode, the pipeline executes the following automated steps:

1. **Mesh Preprocessing**: Conservatively cleans the input mesh, welds duplicated seam vertices, removes invalid faces, and subdivides large/long-edge faces when the mesh is too sparse for geometry optimization.
2. **Multi-view Normal Rendering**: Renders the preprocessed coarse 3D mesh into multi-view normal maps (configurable between 4 or 6 views: front, back, left, right, and optionally top, bottom), merged into a single layout.
3. **AI-Driven Detail Generation**: Uses generative AI (e.g., Gemini) to add rich, photorealistic, and structurally coherent details to the multi-view normal maps based on the object's category.
4. **Background Removal & Processing**: Applies background removal and optional Super Resolution (SR) to the generated AI normal maps to ensure clean and high-quality reference signals.
5. **View Splitting**: Separates the merged layout back into distinct directional views, accommodating the choice of 4 or 6 views.
6. **Geometry Optimization (`mv_refine.py`)**:
   - Optimizes the 3D mesh geometry to align with the AI-generated normal maps via differentiable rendering.
   - Utilizes **Frequency Separation** to balance between injecting high-frequency micro-details and allowing macro-contour/silhouette deformations.
7. **Post-Processing Smoothing**: Optionally applies filtering techniques like Taubin smoothing (via `smooth_mesh.py`) to reduce generated surface noise while retaining structural volume.

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
- `code/mesh_sanitize.py`: Conservative mesh preprocessing utilities used before rendering/refinement.
- `code/mv_refine.py`: The core PyTorch-based 3D multi-view geometry optimization engine.
- `code/render_init_normals_merge.py`: Script for rendering initial multi-view normals.
- `code/api_gemini_gen_img.py`: Interface for generating enhanced normal maps using the Gemini API.
- `code/rb_img.py`: Background removal utility.
- `code/smooth_mesh.py`: Post-processing utility for applying smoothing algorithms to the refined mesh.

## Usage

You can run the Python script directly using command-line arguments to optimize a single mesh.

### 1. Set Gemini API Key

Set `GEMINI_API_KEY` in your environment to enable AI-driven normal map enhancement. Optionally set `GEMINI_API_URL` to override the default endpoint.

### 2. Using Command Line

Run the pipeline by providing the necessary arguments to `run_pipeline.py`:

```bash
# Default Parallel Multi-view Mode (4 views by default)
python run_pipeline.py --mesh_path ./data/test_cases/chair_001/concept_mesh.obj --cat chair

# Parallel Mode with 6 views and Style Reference
python run_pipeline.py --mesh_path ./data/test_cases/car_001/concept_mesh.glb --cat car --num_views 6 --style_ref ./path/to/style_image.jpg

# Autoregressive Mode (progressively generates and optimizes)
python run_pipeline.py --mesh_path ./data/test_cases/chair_001/concept_mesh.obj --cat chair --autoregressive --n_azimuth 12
```

**Available Arguments in `run_pipeline.py`**:

- `--mesh_path`: **(Required)** Path to the input coarse 3D mesh file (`.obj` or `.glb`).
- `--out_dir`: Directory to save the outputs. If not provided, it defaults to the directory of the input mesh.
- `--cat`: Category of the object (e.g., `chair`, `car`, `bottle`). This guides the generative AI to produce appropriate details (default: `object`).
- `--re_gen`: Flag to force re-generation of AI images and background removal.
- `--re_remesh`: Flag to force re-execution of the 3D mesh refinement step (`mv_refine.py`).
- `--sr`: Flag to **enable** Super Resolution on the AI-generated normal maps (default is off).
- `--use_frequency_separation`: Flag to enable frequency separation. This blends geometry frequencies to construct high-frequency micro-details while preserving the overall clean global shape/macro-silhouette from the original coarse mesh (default: `True`).
- `--num_views`: Choose whether to use `4` or `6` views for normal generation and mesh refinement in parallel mode (default: `4`).
- `--smooth`: Flag to optionally apply Taubin smoothing directly after the mesh refinement step to denoise the final model while preserving object geometric volume.
- `--smooth_iter`: Controls the number of iterations for Taubin smoothing (default: `10`). Only active when `--smooth` is provided.
- `--autoregressive`: Flag to enable Autoregressive processing mode instead of the default Parallel Multi-view mode. Progressively generates and optimizes the mesh view-by-view.
- `--n_azimuth`: Controls the number of azimuth views used during Autoregressive refinement (default: `12`). Only active when `--autoregressive` is provided.
- `--style_ref`: Optional path to a style reference image to guide the Gemini generation in both Autoregressive and Parallel modes.
