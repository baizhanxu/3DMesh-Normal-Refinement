import requests
import base64
import os

api_url = "http://apicz.boyuerichdata.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
api_key = "sk-eeefxofTRrmeDAat4USfVZE2Ez6uuS1fN95ZY38TisPMsWqm"

# 官网要求：使用 1:1 宽高比，2K 分辨率，以对话方式进行迭代
aspect_ratio = "1:1"
resolution = "2K"

def get_generation_config():
    return {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {
            "aspectRatio": aspect_ratio,
            "imageSize": resolution
        }
    }
    
def process_single_image(input_file, cat="object", num_views=4, style_ref=None):
    if not os.path.exists(input_file):
        print(f"Skipping: {input_file} (Not found)")
        return

    case_dir = os.path.dirname(input_file)
    output_file = os.path.join(case_dir, "gemini_gen_merged_normals.png")
    
    # 避免对已经生成的图片重复请求（如果有需要覆盖可注释掉这段）
    if os.path.exists(output_file):
        print(f"Output already exists: {output_file}, skipping.")
        return

    print(f"Processing single file: {input_file} | Category: {cat} | Views: {num_views}")
    
    try:
        with open(input_file, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()

        view_text = "前后左右及上下六个" if num_views == 6 else "前后左右四个"
        prompt_text = (
            f"我上传的图片为 {cat} 物体{view_text}视角的法线图，请你对该图片进行编辑，要求在保持物体几何形状与轮廓不变的情况下，为该法线图添加符合 {cat} 物体类别的真实、清晰的几何法线细节，不要添加多余的噪声，不要破坏物体原本的形状与轮廓。"
            f"我希望在法线图中添加美观的 {cat} 法线细节，可以增加一些花纹、条纹或者凹凸作为装饰，请确保多视角之间具有一致性，没有互相矛盾的地方。"
            "注意：严格保持法线图中物体各部分的几何形状与轮廓不变，布局与输入的法线图完全一致，生成的图片使用黑色背景。"
        )

        parts = [
            {"text": prompt_text},
            {"inline_data": {"mime_type": "image/png", "data": b64_data}}
        ]
        
        if style_ref and os.path.exists(style_ref):
            print(f"Using style reference: {style_ref}")
            with open(style_ref, "rb") as f:
                style_b64 = base64.b64encode(f.read()).decode()
            ext_str = "jpeg" if style_ref.lower().endswith("jpg") or style_ref.lower().endswith("jpeg") else "png"
            parts.extend([
                {"text": "【风格参考图】: 此外，我提供了一张风格参考图片，请你将这张图片中所展示的物体外观风格提取，并且通过法线图展现出来，应用到生成的法线图中。但需要严格保证物体的形状和轮廓不变。"},
                {"inline_data": {"mime_type": f"image/{ext_str}", "data": style_b64}}
            ])

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": get_generation_config()
        }
        # 可以增加一些花纹、条纹或者凹凸作为装饰
        try:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=300  # increased timeout slightly just in case
            )
            response.raise_for_status() # Raise exception for bad status codes like 500, 502, 504 etc
        except requests.exceptions.Timeout:
            print("Error: Gemini API request timed out after 120 seconds.")
            return False
        except requests.exceptions.HTTPError as err:
             print(f"Error: Gemini API request HTTP error: {err}")
             print(f"Response content: {response.content}")
             return False
        except requests.exceptions.RequestException as e:
            print(f"Error: Gemini API request failed: {e}")
            return False
        
        # 处理可能返回的非 JSON (报错网页/502等)
        try:
            res_json = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"API Error - Cannot parse JSON. Status Code: {response.status_code}")
            print(f"Raw Response: {response.text[:200]}...")
            return
            
        # Check for errors in response
        if "error" in res_json:
            print(f"API Error for {input_file}: {res_json['error']}")
            return
            
        for part in res_json["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                ext = ".jpg" if "jpeg" in part["inlineData"]["mimeType"] else ".png"
                out_path = os.path.join(case_dir, f"gemini_gen_merged_normals{ext}")
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(part["inlineData"]["data"]))
                print(f"Saved: {out_path}")
                break
    except Exception as e:
        print(f"Failed processing {input_file}: {e}")

if __name__ == "__main__":
    # 你可以在这里指定想处理的某一张法线图绝对路径，和它的类型
    # example_input = "/Users/jmb/Downloads/aaa-important/prog_data/chair/case_1/merged_view_normals.png"
    # process_single_image(example_input, cat="chair")
    import sys
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        cat_name = sys.argv[2] if len(sys.argv) > 2 else "object"
        process_single_image(target_file, cat=cat_name)
    else:
        print("Usage: python gen_text_cond_single.py <path_to_png> [category_name]")

def process_sr_image(input_file):
    """
    Call API to perform Super Resolution on a single view image.
    """
    if not os.path.exists(input_file):
        print(f"Skipping SR: {input_file} (Not found)")
        return False

    case_dir = os.path.dirname(input_file)
    basename = os.path.basename(input_file)
    name, ext = os.path.splitext(basename)
    
    # 根据 API 输出习惯，可能生成的是 .png 或 .jpg
    output_file_png = os.path.join(case_dir, f"{name}_sr.png")
    output_file_jpg = os.path.join(case_dir, f"{name}_sr.jpg")
    
    if os.path.exists(output_file_png):
        print(f"SR Output already exists: {output_file_png}, skipping.")
        return output_file_png
    if os.path.exists(output_file_jpg):
        print(f"SR Output already exists: {output_file_jpg}, skipping.")
        return output_file_jpg

    print(f"Processing SR for: {input_file}")
    
    # URL 保持原来的基准格式，按照用户意图如果是 nano-banana-pro 我们先尝试替换模型名称
    sr_api_url = api_url
    try:
        with open(input_file, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()

        payload = {
            "contents": [{
                "parts": [
                    {"text": "请提高这张图片的分辨率，保持原本的形状和色彩分布，只做清晰度和超分辨率提升。"},
                    {"inline_data": {"mime_type": f"image/{ext.strip('.')}", "data": b64_data}}
                ],
                "generationConfig": get_generation_config()
            }]
        }
        
        try:
            response = requests.post(
                sr_api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            # 如果 nano-banana-pro 模型名不可用，fallback 到原有的模型
            if response.status_code == 404:
                response = requests.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: SR API request failed: {e}")
            return False
            
        res_json = response.json()
        if "error" in res_json:
            print(f"API Error for {input_file}: {res_json['error']}")
            return False
            
        for part in res_json["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                out_ext = ".jpg" if "jpeg" in part["inlineData"]["mimeType"] else ".png"
                out_path = os.path.join(case_dir, f"{name}_sr{out_ext}")
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(part["inlineData"]["data"]))
                print(f"Saved SR image: {out_path}")
                return out_path
    except Exception as e:
        print(f"Failed SR processing {input_file}: {e}")
        return False


def process_inpainting_image(coarse_file, mask_file, output_file, cat="object", style_ref=None):
    if not os.path.exists(coarse_file):
        print(f"Skipping: {coarse_file} (Not found)")
        return False

    if os.path.exists(output_file):
        print(f"Output already exists: {output_file}, skipping.")
        return True

    print(f"Processing inpainting for: {coarse_file} | Mask: {mask_file} | Category: {cat}")
    
    try:
        with open(coarse_file, "rb") as f:
            coarse_b64 = base64.b64encode(f.read()).decode()
            
        contents = []

        # Round 1 (对话方式迭代第一轮): 传入初始粗糙网格
        contents.append({
            "role": "user",
            "parts": [
                {"text": f"这是 {cat} 物体的初始三维法线渲染图。它提供了基本的几何轮廓结构。"},
                {"inline_data": {"mime_type": "image/png", "data": coarse_b64}}
            ]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "我已了解该物体的基本三维法线结构。接下来的修改中，我会在保留主体结构的基础上，允许对局部的几何轮廓和边缘进行小幅度的合理调整和优化，使其更符合真实物体或风格设定。这部分生成的背景必须是纯黑色。"}]
        })

        # Round 2 (对话方式迭代第二轮): 传入遮罩进行局部重绘
        if os.path.exists(mask_file):
            with open(mask_file, "rb") as f:
                mask_b64 = base64.b64encode(f.read()).decode()

            style_text = ""
            if style_ref and os.path.exists(style_ref):
                style_text = "保留原始形状的同时，请参考图片以应用风格。"

            # 官网 Prompt 语法
            mask_prompt = (
                f"Using the provided mask image (where the white boxes represent unseen/missing regions), "
                f"fill the white masked regions with naturally generated 3D normal map details. "
                f"You are encouraged to slightly refine and adjust the contours and edges in the generated regions (for example, smoothing sharp or unnatural corners) to make the overall shape look more realistic and stylistically coherent. The background must remain pure black."
            )

            parts2 = [
                {"text": mask_prompt},
                {"inline_data": {"mime_type": "image/png", "data": mask_b64}}
            ]
            
            if style_ref and os.path.exists(style_ref):
                with open(style_ref, "rb") as f:
                    style_b64 = base64.b64encode(f.read()).decode()
                ext_str = "jpeg" if style_ref.lower().endswith("jpg") or style_ref.lower().endswith("jpeg") else "png"
                parts2.extend([
                    {"text": "【风格参考图】: 此外，我提供了一张风格参考图片。请你将这张图片中所展示的物体外观风格应用到生成的法线图中。允许小幅度修改物体的轮廓和边缘（例如将生硬的直角变得平滑），使其更贴近参考图片的物理形貌和质感。背景务必保持严格纯黑。"},
                    {"inline_data": {"mime_type": f"image/{ext_str}", "data": style_b64}}
                ])
                
            contents.append({
                "role": "user",
                "parts": parts2
            })
            
        payload = {
            "contents": contents,
            "generationConfig": get_generation_config()
        }
        
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        
        try:
            res_json = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"API Error - Cannot parse JSON. Status Code: {response.status_code}")
            return False
            
        if "error" in res_json:
            print(f"API Error for {coarse_file}: {res_json['error']}")
            return False
            
        for part in res_json.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if "inlineData" in part:
                with open(output_file, "wb") as f:
                    f.write(base64.b64decode(part["inlineData"]["data"]))
                print(f"Saved Inpainted img: {output_file}")
                return True
                
        return False
    except Exception as e:
        print(f"Failed inpainting processing {coarse_file}: {e}")
        return False