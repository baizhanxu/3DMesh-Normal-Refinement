import os
import sys
import subprocess
from PIL import Image
from pathlib import Path

# Add 'code' directory to sys.path so we can import modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, "code"))

try:
    from render_init_normals_merge import process_single_mesh, split_image_into_six_views
    from api_gemini_gen_img import process_single_image as gemini_process
    from api_gemini_gen_img import process_sr_image
    from rb_img import process_single_image as rb_process
    from smooth_mesh import smooth_taubin
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

import argparse

def split_image_into_vier_views(merged_path, save_dir):
    """
    Split the 2x2 merged normal map back into 4 views
    """
    import numpy as np
    img = Image.open(merged_path)
    
    # # 将模型生成的法线图的红色通道（X轴）反转，以对齐渲染图的 OpenGL/DirectX 标准（解决右红左绿变反的问题）
    # img_np = np.array(img)
    # if img_np.shape[-1] >= 3:
    #     img_np[:, :, 0] = 255 - img_np[:, :, 0]
    # img = Image.fromarray(img_np)

    w, h = img.size
    
    half_w = w // 2
    half_h = h // 2
    
    views = [
        (0, 0, half_w, half_h),           # view1 (0) top-left
        (half_w, 0, w, half_h),           # view2 (1) top-right
        (0, half_h, half_w, h),           # view3 (2) bottom-left
        (half_w, half_h, w, h)            # view4 (3) bottom-right
    ]
    
    output_files = []
    for i, bbox in enumerate(views):
        view_img = img.crop(bbox)
        # mv_refine treats these as views 0,1,2,3 internally corresponding to what mv gives
        out_path = os.path.join(save_dir, f"view{i+1}.png")
        view_img.save(out_path)
        output_files.append(out_path)
    
    print(f"Split {merged_path} into 4 views in {save_dir}")
    return output_files

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Refinement Pipeline")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input mesh file (.obj or .glb)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save the outputs. Defaults to the directory of the input mesh.")
    parser.add_argument("--cat", type=str, default="object", help="Category of the object (e.g., chair, car) to guide the detail generation.")
    parser.add_argument("--re_gen", action="store_true", help="Force re-generation of normals and segmentation")
    parser.add_argument("--re_remesh", action="store_true", help="Force re-execution of the mesh refinement step")
    parser.add_argument("--sr", action="store_true", help="Enable super resolution before optimization")
    parser.add_argument("--use_frequency_separation", action="store_true", help="Enable frequency separation to preserve global shape", default=True)
    parser.add_argument("--force_subdivide", action="store_true", help="Force subdivide the initial mesh to ensure higher vertex count for sparse meshes", default=False)
    parser.add_argument("--num_views", type=int, choices=[4, 6], default=4, help="Choose whether to use 4 views or 6 views for refinement.")
    parser.add_argument("--smooth", action="store_true", help="Apply Taubin smoothing after mesh refinement")
    parser.add_argument("--smooth_iter", type=int, default=10, help="Number of Taubin smoothing iterations")
    parser.add_argument("--autoregressive", action="store_true", help="Enable Autoregressive processing mode instead of parallel Multi-view mode")
    parser.add_argument("--n_azimuth", type=int, default=12, help="Number of azimuth views for Autoregressive refinement")
    parser.add_argument("--style_ref", type=str, default=None, help="Optional path to a style reference image for both Autoregressive and Parallel modes.")
    args = parser.parse_args()

    obj_path = Path(args.mesh_path)
    if not obj_path.exists():
        print(f"Mesh file {args.mesh_path} does not exist!")
        return

    if args.out_dir is None:
        case_dir = obj_path.parent
    else:
        case_dir = Path(args.out_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

    mesh_name = obj_path.stem
    re_gen = args.re_gen
    re_remesh = args.re_remesh
    use_sr = args.sr
    use_freq_sep = args.use_frequency_separation
    force_subdivide = args.force_subdivide
    num_views = args.num_views
    apply_smooth = args.smooth
    smooth_iter = args.smooth_iter
    cat_name = args.cat

    print(f"==========================================")
    print(f"Processing mesh: {obj_path.name} in {case_dir}")
    print(f"==========================================")

    # 0. 对 mesh 进行 subdivide 操作（如果用户指定了 force_subdivide 或者检测到顶点数量过少）
    import trimesh
    mesh = trimesh.load(str(obj_path), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        
    original_vertex_count = len(mesh.vertices)
    target_vertex_count = 10000
    if force_subdivide:
        target_vertex_count = max(10000, original_vertex_count * 4)

    subdivided_obj_path = case_dir / f"{mesh_name}_subdivided.obj"
       

    if original_vertex_count < target_vertex_count:
        if not subdivided_obj_path.exists() or re_gen:
            print(f"[Step 0] Remeshing for uniform vertex distribution... Initial vertices: {original_vertex_count}")
            sdf_resolution = 128
            sdf_padding_ratio = 0.05
            print(f"         └─ Using SDF remeshing (resolution={sdf_resolution}, padding_ratio={sdf_padding_ratio})")
            mesh = remesh_via_sdf(
                mesh,
                target_vertex_count=target_vertex_count,
                sdf_resolution=sdf_resolution,
                sdf_padding_ratio=sdf_padding_ratio,
            )
            
            mesh.export(str(subdivided_obj_path))
            print(f"[Step 0] Uniformly subdivided mesh saved to {subdivided_obj_path.name} with {len(mesh.vertices)} vertices.")
        else:
            print(f"[Step 0] Skipped. Subdivided mesh {subdivided_obj_path.name} already exists.")
        
        # 更新 obj_path，让后续的渲染和优化都使用这个细分过的模型
        obj_path = subdivided_obj_path
    else:
        print(f"[Step 0] Skipped. Mesh has {original_vertex_count} vertices, which is sufficient.")     
    if args.autoregressive:
        # Autoregressive block
        n_views = args.n_azimuth
        print(f"==== Autoregressive optimization for {case_dir.name} ====")
        
        current_mesh_path = obj_path
        historical_img_paths = []
        historical_views = []
        
        for v_idx in range(n_views):
            print(f"\n--- [AR Step {v_idx+1}/{n_views}] View angle index: {v_idx} ---")
            view_dir = case_dir / f"view_step_{v_idx}"
            view_dir.mkdir(exist_ok=True)
            
            rendered_normal_path = view_dir / f"rendered_normal_{v_idx}.png"
            mask_path = view_dir / "visible_mask.png"
            inpainted_normal_path = view_dir / "inpainted_normal.png"
            segmented_inpainted_path = view_dir / "inpainted_normal_segmented.png"
            next_mesh_path = view_dir / f"refined_step_{v_idx}.obj"
            
            historical_img_paths.append(str(segmented_inpainted_path))
            historical_views.append(str(v_idx))
            
            if next_mesh_path.exists() and not re_gen and not re_remesh:
                print(f"[Skip] {next_mesh_path.name} already exists.")
                current_mesh_path = next_mesh_path
                continue
                
            coarse_normal_path = view_dir / f"coarse_normal_{v_idx}.png"
            if not coarse_normal_path.exists() or re_gen:
                print("AR-0. Rendering coarse mesh for view as geometric reference...")
                cmd = [
                    "python", os.path.join(curr_dir, "code", "mv_autoregressive_step.py"),
                    "--name", case_dir.name,
                    "--mesh_path", str(obj_path),
                    "--autoreg_mode", "render",
                    "--view_idx", str(v_idx),
                    "--n_azimuth", str(n_views),
                    "--out_dir", str(view_dir),
                    "--force_normalize"
                ]
                subprocess.run(cmd, check=True)
                if rendered_normal_path.exists():
                    os.rename(rendered_normal_path, coarse_normal_path)
            
            print("AR-1. Rendering current mesh and extracting new-part mask...")
            cmd_curr = [
                "python", os.path.join(curr_dir, "code", "mv_autoregressive_step.py"),
                "--name", case_dir.name,
                "--mesh_path", str(current_mesh_path),
                "--autoreg_mode", "render",
                "--view_idx", str(v_idx),
                "--n_azimuth", str(n_views),
                "--out_dir", str(view_dir),
                "--out_mask_path", str(mask_path)
            ]
            if current_mesh_path == obj_path:
                cmd_curr.append("--force_normalize")
                
            subprocess.run(cmd_curr, check=True)
            
            print("AR-2. Inpainting new regions using Gemini...")
            from api_gemini_gen_img import process_inpainting_image
            
            success = False
            max_retries = 3
            for attempt in range(max_retries):
                success = process_inpainting_image(
                    coarse_file=str(coarse_normal_path),
                    mask_file=str(mask_path),
                    output_file=str(inpainted_normal_path),
                    cat=cat_name,
                    style_ref=args.style_ref
                )
                if success:
                    break
                print(f"Gemini inpainting failed. Retrying ({attempt + 1}/{max_retries})...")
                import time
                time.sleep(2)
            
            # Fallback handling in case of API failure
            if not success:
                print(f"Gemini inpainting failed after {max_retries} attempts. Falling back to the original render...")
                import shutil
                if rendered_normal_path.exists():
                    shutil.copy(str(rendered_normal_path), str(inpainted_normal_path))
                else:
                    shutil.copy(str(coarse_normal_path), str(inpainted_normal_path))
            
            print("AR-3. Removing background from inpainted normal...")
            rb_process(str(inpainted_normal_path))
            
            print(f"AR-4. Optimizing mesh (joint optimization with {len(historical_views)} views)...")
            cmd_opt = [
                "python", os.path.join(curr_dir, "code", "mv_autoregressive_step.py"),
                "--name", case_dir.name,
                "--mesh_path", str(current_mesh_path),
                "--autoreg_mode", "optimize",
                "--view_idx", str(v_idx),
                "--n_azimuth", str(n_views),
                "--out_dir", str(view_dir),
                "--output_mesh_path", str(next_mesh_path),
                "--views", *historical_views,
                "--img_paths", *historical_img_paths,
                "--use_frequency_separation"
            ]
            if current_mesh_path == obj_path:
                cmd_opt.append("--force_normalize")
                
            subprocess.run(cmd_opt, check=True)
            current_mesh_path = next_mesh_path
            
        print(f"[Done] Final AR mesh saved at {current_mesh_path}")
        
        if apply_smooth:
            smoothed_obj_path = case_dir / f"{mesh_name}_autoreg_smoothed.obj"
            if not smoothed_obj_path.exists() or re_remesh:
                print(f"[AR Step 6] Applying Taubin smoothing for {smooth_iter} iterations...")
                smooth_taubin(str(current_mesh_path), str(smoothed_obj_path), smooth_iter)
    else:
        # 1. 渲染多视角法线并合并
        merged_normals_path = case_dir / "merged_view_normals.png"
        if not merged_normals_path.exists() or re_gen:
            print(f"[Step 1] Rendering initial multi-view normals for {obj_path.name}...")
            success = process_single_mesh(str(obj_path), str(case_dir), num_views=num_views)
            if not success:
                print(f"Failed to render normals for {case_dir.name}")
                return
        else:
            print(f"[Step 1] Skipped. {merged_normals_path.name} already exists.")

        # 2. Gemini API 生成添加细节的法线图
        gemini_output_path = case_dir / "gemini_gen_merged_normals.png"
        # 也有可能会存为 .jpg，为了严谨最好都判断一下
        gemini_output_path_alt = case_dir / "gemini_gen_merged_normals.jpg"
        
        has_gemini = gemini_output_path.exists() or gemini_output_path_alt.exists()
        if not has_gemini or re_gen:
            print(f"[Step 2] Applying Gemini to generate detailed normals...")
            gemini_process(str(merged_normals_path), cat=cat_name, num_views=num_views, style_ref=args.style_ref)
        else:
            print(f"[Step 2] Skipped. Gemini output already exists.")

        # 确定实际存在的 Gemini 输出文件
        actual_gemini_path = None
        if gemini_output_path.exists(): actual_gemini_path = gemini_output_path
        elif gemini_output_path_alt.exists(): actual_gemini_path = gemini_output_path_alt

        if not actual_gemini_path:
            print(f"API generation failed or missing output for {case_dir.name}")
            return

        # 3. 去除背景
        segmented_path = case_dir / f"{actual_gemini_path.stem}_segmented.png"
        if not segmented_path.exists() or re_gen:
            print(f"[Step 3] Removing background from generated normals...")
            rb_process(str(actual_gemini_path))
        else:
            print(f"[Step 3] Skipped. {segmented_path.name} already exists.")

        if not segmented_path.exists():
            print(f"Background removal failed for {case_dir.name}")
            return

        # 4. 把合并图拆分为对应数量的 view
        views_exist = all((case_dir / f"view{i}.png").exists() for i in range(1, num_views + 1))
        view_paths = [str(case_dir / f"view{i}.png") for i in range(1, num_views + 1)]
        
        if not views_exist or re_gen:
            print(f"[Step 4] Splitting segmented image into {num_views} views...")
            if num_views == 6:
                split_image_into_six_views(str(segmented_path), str(case_dir))
            else:
                split_image_into_vier_views(str(segmented_path), str(case_dir))
        else:
            print(f"[Step 4] Skipped. Separated views already exist.")

        # 5. mesh 细化
        refined_obj_path = case_dir / f"{mesh_name}_refined_new.obj"
        if not refined_obj_path.exists() or re_gen or re_remesh:
            print(f"[Step 5] Running MV Refine to optimize mesh...")
            
            two_view_0_2_cats = ["globe", "mug"]
            two_view_1_3_cats = ["bottle", "dispenser", "kettle", "pot", "keyboard"]

            if cat_name in two_view_1_3_cats:
                if num_views == 6:
                    selected_views = ["1", "3", "4", "5"]
                    selected_img_paths = [view_paths[1], view_paths[3], view_paths[4], view_paths[5]]
                    print(f"[{cat_name}] is a specific category, using 4 views (1, 3, 4, 5).")
                else:
                    selected_views = ["1", "3"]
                    selected_img_paths = [view_paths[1], view_paths[3]]
                    print(f"[{cat_name}] is a specific category, using 2 views (1, 3).")
            elif cat_name in two_view_0_2_cats:
                if num_views == 6:
                    selected_views = ["0", "2", "4", "5"]
                    selected_img_paths = [view_paths[0], view_paths[2], view_paths[4], view_paths[5]]
                    print(f"[{cat_name}] is a specific category, using 4 views (0, 2, 4, 5).")
                else:
                    selected_views = ["0", "2"]
                    selected_img_paths = [view_paths[0], view_paths[2]]
                    print(f"[{cat_name}] is a specific category, using 2 views (0, 2).")
            else:
                print(f"[{cat_name}] is a general category, using all {num_views} views for refinement.")
                if num_views == 6:
                    selected_views = ["0", "1", "2", "3", "4", "5"]
                else:
                    selected_views = ["0", "1", "2", "3"]
                selected_img_paths = view_paths
                
            if use_sr:
                print(f"[Step 4.5] Checking Super Resolution and Background Removal for selected views...")
                sr_img_paths = []
                for img_path in selected_img_paths:
                    # 1. 尝试超分辨率
                    sr_out = process_sr_image(img_path)
                    target_image_for_refine = sr_out if sr_out else img_path
                        
                    # 2. 对超分辨率（或原图）进行去背景操作
                    target_base = Path(target_image_for_refine).stem
                    target_dir = Path(target_image_for_refine).parent
                    segmented_out = target_dir / f"{target_base}_segmented.png"
                    
                    if not segmented_out.exists() or re_gen:
                        print(f"Removing background for: {target_image_for_refine}")
                        rb_res = rb_process(str(target_image_for_refine))
                        if rb_res:
                            sr_img_paths.append(str(segmented_out))
                        else:
                            print(f"Failed background removal on {target_image_for_refine}, using it directly.")
                            sr_img_paths.append(target_image_for_refine)
                    else:
                        print(f"Segmented image already exists for: {target_base}")
                        sr_img_paths.append(str(segmented_out))

                selected_img_paths = sr_img_paths

            # We call mv_refine.py via subprocess to ensure clean GPU memory and independent args
            cmd = [
                "python", os.path.join(curr_dir, "code", "mv_refine.py"),
                "--name", case_dir.name,
                "--mesh_path", str(obj_path),
                "--views", *selected_views,
                "--img_paths", *selected_img_paths,
                "--out_dir", str(case_dir),
                "--output_mesh_path", str(refined_obj_path),
            ]
            
            if use_freq_sep:
                cmd.append("--use_frequency_separation")
            if force_subdivide:
                cmd.append("--force_subdivide")

            try:
                subprocess.run(cmd, check=True)
                print(f"[Success] Refined mesh saved at {refined_obj_path.name}")
                
                # 6. 可选去噪步骤
                if apply_smooth:
                    smoothed_obj_path = case_dir / f"{mesh_name}_refined_smoothed.obj"
                    print(f"[Step 6] Option enabled. Applying Taubin smoothing for {smooth_iter} iterations...")
                    smooth_taubin(str(refined_obj_path), str(smoothed_obj_path), smooth_iter)
                    
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to refine mesh for {case_dir.name}: {e}")
        else:
            print(f"[Step 5] Skipped. {refined_obj_path.name} already exists.")
            
            if apply_smooth:
                smoothed_obj_path = case_dir / f"{mesh_name}_refined_smoothed.obj"
                if not smoothed_obj_path.exists():
                    print(f"[Step 6] Option enabled. Applying Taubin smoothing for {smooth_iter} iterations...")
                    smooth_taubin(str(refined_obj_path), str(smoothed_obj_path), smooth_iter)
                else:
                    print(f"[Step 6] Skipped. Smoothed mesh {smoothed_obj_path.name} already exists.")

if __name__ == "__main__":
    main()
