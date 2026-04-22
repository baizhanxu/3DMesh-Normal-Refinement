import os
import torch
import pathlib
from PIL import Image
import numpy as np

# Adjust imports to be relative if needed, or keeping them absolute if they are in PYTHONPATH
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)
# Optional: if you rely on an external continuous-remeshing repository, uncomment and set the correct path
# sys.path.append(os.path.abspath(os.path.join(curr_dir, "../../continuous-remeshing")))

from core.remesh import calc_vertex_normals
from core.opt import MeshOptimizer
from util.func import load_obj, make_star_cameras, normalize_vertices, save_images
from util.render import NormalsRenderer

def merge_four_views(base_dir, output_name="merged_view_normals.png"):
    view_names = ["init_normals/00.png", "init_normals/01.png", "init_normals/02.png", "init_normals/03.png"]
    images = []
    print(f"Loading from {base_dir} for merge...")
    for name in view_names:
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            print(f"Error: Not found {path}")
            return False
        images.append(Image.open(path))

    width, height = images[0].size
    for i, img in enumerate(images):
        if img.size != (width, height):
            images[i] = img.resize((width, height), Image.Resampling.LANCZOS)

    merged_img = Image.new('RGB', (width * 2, height * 2), (255, 255, 255))
    merged_img.paste(images[0], (0, 0))
    merged_img.paste(images[1], (width, 0))
    merged_img.paste(images[2], (0, height))
    merged_img.paste(images[3], (width, height))

    output_path = os.path.join(base_dir, output_name)
    merged_img.save(output_path)
    print(f"Merge successful: {output_path}")
    return True

def merge_six_views(base_dir, output_name="merged_view_normals.png"):
    view_names = [f"init_normals/{i:02d}.png" for i in range(6)]
    images = []
    print(f"Loading from {base_dir} for merge...")
    for name in view_names:
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            print(f"Error: Not found {path}")
            return False
        images.append(Image.open(path))

    width, height = images[0].size
    
    merged_img = Image.new('RGB', (width * 3, height * 2), (255, 255, 255))
    
    positions = [
        (0, 0),         # view1
        (width, 0),     # view2
        (width*2, 0),   # view3
        (0, height),    # view4
        (width, height),# view5 (Top)
        (width*2, height) # view6 (Bottom)
    ]
    
    for img, pos in zip(images, positions):
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        merged_img.paste(img, pos)

    output_path = os.path.join(base_dir, output_name)
    merged_img.save(output_path)
    print(f"6-view Merge successful: {output_path}")
    return True

def split_image_into_six_views(merged_path, save_dir):
    """
    Split the 3x2 merged normal map back into 6 views.
    Handles extra width if image was padded.
    """
    import numpy as np
    img = Image.open(merged_path)
    
    # 将模型生成的法线图的红色通道（X轴）反转
    img_np = np.array(img)
    if img_np.shape[-1] >= 3:
        img_np[:, :, 0] = 255 - img_np[:, :, 0]
    img = Image.fromarray(img_np)

    w, h = img.size
    
    # Image might have been padded to square by API. The original shape was 3x2 ratio.
    # original_w / original_h = 3/2 -> original_h = original_w * 2 / 3
    if w == h:
        # It's padded. Assuming padding is at the bottom.
        actual_h = int(w * 2 / 3)
        img = img.crop((0, 0, w, actual_h))
        h = actual_h

    cell_w = w // 3
    cell_h = h // 2
    
    # Actually the script creates 3x2 with size (width*3, height*2)
    # The cell should be square as it was originally.
    
    actual_size = min(cell_w, cell_h)
    
    # We arranged them as:
    # 0, 1, 2
    # 3, 4, 5
    # For mv_refine, views from init are 0,1,2,3 (front right back left), 4 (top), 5 (bottom)
    file_name_mapping = {
        0: 1, # view1 -> mv view0
        1: 2, # view2 -> mv view1
        2: 3, # view3 -> mv view2
        3: 4, # view4 -> mv view3
        4: 5, # view5 -> mv view4
        5: 6  # view6 -> mv view5
    }
    
    views = [
        (0, 0, actual_size, actual_size),                           # view1
        (cell_w, 0, cell_w + actual_size, actual_size),             # view2
        (cell_w * 2, 0, cell_w * 2 + actual_size, actual_size),     # view3
        (0, cell_h, actual_size, cell_h + actual_size),             # view4
        (cell_w, cell_h, cell_w + actual_size, cell_h + actual_size), # view5
        (cell_w * 2, cell_h, cell_w * 2 + actual_size, cell_h + actual_size) # view6
    ]
    
    output_files = []
    for i, bbox in enumerate(views):
        view_img = img.crop(bbox)
        # Crop out the text area if needed to make it square again
        view_img = view_img.crop((0, 0, actual_size, actual_size))
        
        real_id = file_name_mapping[i]
        out_path = os.path.join(save_dir, f"view{real_id}.png")
        view_img.save(out_path)
        output_files.append(out_path)
    
    output_files.sort() # Ensure they are sorted view1 -> view6
    print(f"Split {merged_path} into 6 views in {save_dir}")
    return output_files

def process_single_mesh(obj_file, output_folder, resolution=2048, num_views=4):
    os.makedirs(output_folder, exist_ok=True)
    ref_vertices, ref_faces = load_obj(str(obj_file))
    ref_vertices = normalize_vertices(ref_vertices)
    ref_vertices[..., [0, 2]] = - ref_vertices[..., [0, 2]]  # Match orientation with mv_refine.py

    distance = 10.0
    mv, proj = make_star_cameras(4, 1, distance=distance)

    def make_view(th, ph, device):
        th = torch.tensor(th, device=device)
        ph = torch.tensor(ph, device=device)
        
        phi_rot = torch.eye(3, device=device)
        phi_rot[2,2] = ph.cos()
        phi_rot[2,0] = -ph.sin()
        phi_rot[0,2] = ph.sin()
        phi_rot[0,0] = ph.cos()
        
        theta_rot = torch.eye(3, device=device)
        theta_rot[1,1] = th.cos()
        theta_rot[1,2] = -th.sin()
        theta_rot[2,1] = th.sin()
        theta_rot[2,2] = th.cos()
        
        m = torch.eye(4, device=device)
        m[:3,:3] = theta_rot @ phi_rot
        
        # Translation (distance)
        T = torch.eye(4, device=device)
        T[2,3] = -distance
        
        return T @ m

    if num_views == 6:
        # 5: Top, 6: Bottom
        mv_top = make_view(np.pi/2, 0, mv.device).unsqueeze(0)
        mv_bot = make_view(-np.pi/2, 0, mv.device).unsqueeze(0)
        mv_6 = torch.cat([mv, mv_top, mv_bot], dim=0)

    if 'scissors' in str(obj_file):
        A = 4
        P = 1
        C = A * P
        device = mv.device if hasattr(mv, 'device') else 'cuda'
        
        phi = torch.arange(0, A, device=device) * (2 * torch.pi / A)
        phi_rot = torch.eye(3, device=device)[None, None].expand(A, 1, 3, 3).clone()
        phi_rot[:, 0, 2, 2] = phi.cos()
        phi_rot[:, 0, 2, 0] = -phi.sin()
        phi_rot[:, 0, 0, 2] = phi.sin()
        phi_rot[:, 0, 0, 0] = phi.cos()
        
        theta = torch.tensor([30.0 * torch.pi / 180.0], device=device)
        theta_rot = torch.eye(3, device=device)[None, None].expand(1, P, 3, 3).clone()
        theta_rot[0, :, 1, 1] = theta.cos()
        theta_rot[0, :, 1, 2] = -theta.sin()
        theta_rot[0, :, 2, 1] = theta.sin()
        theta_rot[0, :, 2, 2] = theta.cos()
        
        mv = torch.empty((C, 4, 4), device=device)
        mv[:] = torch.eye(4, device=device)
        mv[:, :3, :3] = (theta_rot @ phi_rot).reshape(C, 3, 3)
        
        trans = torch.eye(4, device=device)
        trans[2, 3] = -distance
        mv = trans @ mv
        
        if num_views == 6:
            mv_6 = torch.cat([mv, mv_top, mv_bot], dim=0)

    if num_views == 6:
        mv = mv_6
        
    renderer = NormalsRenderer(mv, proj, [resolution, resolution])
    ref_normals = calc_vertex_normals(ref_vertices, ref_faces)
    ref_images = renderer.render(ref_vertices, ref_normals, ref_faces)

    if ref_images.ndim == 4 and ref_images.shape[3] in [3, 4]:
        ref_images = ref_images.permute(0, 3, 1, 2)
    
    save_images(ref_images, os.path.join(output_folder, 'init_normals'))

    if ref_images.shape[1] >= 3:
        device = ref_images.device
        if isinstance(mv, torch.Tensor):
            view_rots = mv[:, :3, :3].to(device).float()
        else:
            view_rots = torch.from_numpy(mv[:, :3, :3]).to(device).float()
        
        normals_world = ref_images[:, :3, :, :] * 2.0 - 1.0
        B, C, H, W = normals_world.shape
        normals_flat = normals_world.view(B, 3, -1)
        normals_view_flat = torch.bmm(view_rots, normals_flat)
        normals_view = normals_view_flat.view(B, 3, H, W)
        normals_view = torch.nn.functional.normalize(normals_view, dim=1)
        view_normals_rgb = (normals_view + 1.0) / 2.0
        
        if ref_images.shape[1] == 4:
            alpha = ref_images[:, 3:4, :, :]
            ref_images_view = torch.cat([view_normals_rgb, alpha], dim=1)
            if alpha.max() > 0:
                ref_images_view[:, :3, :, :] *= alpha
        else:
            mask = (torch.norm(ref_images, dim=1, keepdim=True) > 0.1).float()
            ref_images_view = view_normals_rgb * mask
    else:
        ref_images_view = ref_images

    if ref_images_view.shape[0] == 4:
        ref_images_view_nhwc = ref_images_view.permute(0, 2, 3, 1)
        save_images(ref_images_view_nhwc, os.path.join(output_folder, 'init_normals'))
        return merge_four_views(output_folder, output_name="merged_view_normals.png")
    elif ref_images_view.shape[0] == 6:
        ref_images_view_nhwc = ref_images_view.permute(0, 2, 3, 1)
        save_images(ref_images_view_nhwc, os.path.join(output_folder, 'init_normals'))
        return merge_six_views(output_folder, output_name="merged_view_normals.png")

    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    process_single_mesh(args.obj_path, args.output_dir)
