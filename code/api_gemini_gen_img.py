import requests
import base64
import os

api_url = "..." # 这里替换成你实际的 Gemini API URL
api_key = "..." # 这里替换成你实际的 API Key

def process_single_image(input_file, cat="object"):
    if not os.path.exists(input_file):
        print(f"Skipping: {input_file} (Not found)")
        return

    case_dir = os.path.dirname(input_file)
    output_file = os.path.join(case_dir, "gemini_gen_merged_normals.png")
    
    # 避免对已经生成的图片重复请求（如果有需要覆盖可注释掉这段）
    if os.path.exists(output_file):
        print(f"Output already exists: {output_file}, skipping.")
        return

    print(f"Processing single file: {input_file} | Category: {cat}")
    
    try:
        with open(input_file, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()

        payload = {
            "contents": [{
                "parts": [
                    {"text": f"我上传的图片为 {cat} 物体前后左右四个视角的法线图，请为该法线图添加真实且清晰的几何法线细节，不要添加多余的噪声。我希望在法线图中添加美观的 {cat} 法线细节，可以增加一些花纹、条纹或者凹凸作为装饰。注意：严格保持法线图中物体各部分的几何形状与轮廓不变，布局与输入的法线图完全一致，生成的图片使用黑色背景，使用最高的分辨率。"},
                    {"inline_data": {"mime_type": "image/png", "data": b64_data}}
                ]
            }]
        }
        # 可以增加一些花纹、条纹或者凹凸作为装饰
        try:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120  # increased timeout slightly just in case
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
                ]
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
