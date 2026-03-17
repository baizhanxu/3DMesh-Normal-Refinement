import os
import sys
import subprocess
from PIL import Image
from pathlib import Path

# Add 'code' directory to sys.path so we can import modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, "code"))

try:
    from render_init_normals_merge import process_single_mesh
    from api_gemini_gen_img import process_single_image as gemini_process
    from api_gemini_gen_img import process_sr_image
    from rb_img import process_single_image as rb_process
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
    
    # 将模型生成的法线图的红色通道（X轴）反转，以对齐渲染图的 OpenGL/DirectX 标准（解决右红左绿变反的问题）
    img_np = np.array(img)
    if img_np.shape[-1] >= 3:
        img_np[:, :, 0] = 255 - img_np[:, :, 0]
    img = Image.fromarray(img_np)

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
    parser.add_argument("--root_dir", type=str, default="./data/test_cases", help="Directory containing the cases")
    parser.add_argument("--mesh_name", type=str, default="concept_mesh", help="Base name of the mesh file (e.g., concept_mesh)")
    parser.add_argument("--re_gen", action="store_true", help="Force re-generation of normals and segmentation")
    parser.add_argument("--re_remesh", action="store_true", help="Force re-execution of the mesh refinement step")
    parser.add_argument("--no_sr", action="store_true", help="Disable super resolution before optimization")
    parser.add_argument("--use_frequency_separation", action="store_true", help="Enable frequency separation to preserve global shape", default=True)
    parser.add_argument("--force_subdivide", action="store_true", help="Force subdivide the initial mesh to ensure higher vertex count for sparse meshes", default=False)
    args = parser.parse_args()

    root_dir = args.root_dir
    mesh_name = args.mesh_name
    re_gen = args.re_gen
    re_remesh = args.re_remesh
    use_sr = not args.no_sr
    use_freq_sep = args.use_frequency_separation
    force_subdivide = args.force_subdivide

    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Root dir {root_dir} does not exist!")
        return

    # Iterate over case directories
    for case_dir in root_path.iterdir():
        if not case_dir.is_dir():
            continue

        obj_path = case_dir / f"{mesh_name}.obj"
        if not obj_path.exists():
            obj_path = case_dir / f"{mesh_name}.glb"
            if not obj_path.exists():
                print(f"Object file not found for case: {case_dir.name}")
                continue

        print(f"==========================================")
        print(f"Processing case: {case_dir.name}")
        print(f"==========================================")

        # 1. 渲染多视角法线并合并
        merged_normals_path = case_dir / "merged_view_normals.png"
        if not merged_normals_path.exists() or re_gen:
            print(f"[Step 1] Rendering initial multi-view normals for {obj_path.name}...")
            success = process_single_mesh(str(obj_path), str(case_dir))
            if not success:
                print(f"Failed to render normals for {case_dir.name}")
                continue
        else:
            print(f"[Step 1] Skipped. {merged_normals_path.name} already exists.")

        # 2. Gemini API 生成添加细节的法线图
        gemini_output_path = case_dir / "gemini_gen_merged_normals.png"
        # 也有可能会存为 .jpg，为了严谨最好都判断一下
        gemini_output_path_alt = case_dir / "gemini_gen_merged_normals.jpg"
        
        has_gemini = gemini_output_path.exists() or gemini_output_path_alt.exists()
        if not has_gemini or re_gen:
            print(f"[Step 2] Applying Gemini to generate detailed normals...")
            # 可以根据文件夹名字提取类别，如 chair_xxxx -> chair
            cat_name = case_dir.name.split('_')[0] 
            gemini_process(str(merged_normals_path), cat=cat_name)
        else:
            print(f"[Step 2] Skipped. Gemini output already exists.")

        # 确定实际存在的 Gemini 输出文件
        actual_gemini_path = None
        if gemini_output_path.exists(): actual_gemini_path = gemini_output_path
        elif gemini_output_path_alt.exists(): actual_gemini_path = gemini_output_path_alt

        if not actual_gemini_path:
            print(f"API generation failed or missing output for {case_dir.name}")
            continue

        # 3. 去除背景
        segmented_path = case_dir / f"{actual_gemini_path.stem}_segmented.png"
        if not segmented_path.exists() or re_gen:
            print(f"[Step 3] Removing background from generated normals...")
            rb_process(str(actual_gemini_path))
        else:
            print(f"[Step 3] Skipped. {segmented_path.name} already exists.")

        if not segmented_path.exists():
            print(f"Background removal failed for {case_dir.name}")
            continue

        # 4. 把合并图拆分为四个 view
        views_exist = all((case_dir / f"view{i}.png").exists() for i in range(1, 5))
        view_paths = [str(case_dir / f"view{i}.png") for i in range(1, 5)]
        
        if not views_exist or re_gen:
            print(f"[Step 4] Splitting segmented image into 4 views...")
            split_image_into_vier_views(str(segmented_path), str(case_dir))
        else:
            print(f"[Step 4] Skipped. Separated views already exist.")

        # 5. mesh 细化
        refined_obj_path = case_dir / f"{mesh_name}_refined_new.obj"
        if not refined_obj_path.exists() or re_gen or re_remesh:
            print(f"[Step 5] Running MV Refine to optimize mesh...")
            
            # 判断类别，如果是圆柱/球体类，则仅使用正背面两个视角 (0和2)
            cat_name = case_dir.name.split('_')[0]
            two_view_0_2_cats = ["globe", "mug"]
            two_view_1_3_cats = ["bottle", "dispenser", "kettle", "pot", "keyboard"]

            if cat_name in two_view_1_3_cats:
                selected_views = ["1", "3"]
                selected_img_paths = [view_paths[1], view_paths[3]]
                print(f"[{cat_name}] is a cylindrical/spherical category, using 2 views (1, 3) instead of 4.")
            elif cat_name in two_view_0_2_cats:
                selected_views = ["0", "2"]
                selected_img_paths = [view_paths[0], view_paths[2]]
                print(f"[{cat_name}] is a cylindrical/spherical category, using 2 views (0, 2) instead of 4.")
            else:
                print(f"[{cat_name}] is not a cylindrical/spherical category, using all 4 views for refinement.")
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
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to refine mesh for {case_dir.name}: {e}")
        else:
            print(f"[Step 5] Skipped. {refined_obj_path.name} already exists.")

if __name__ == "__main__":
    main()
