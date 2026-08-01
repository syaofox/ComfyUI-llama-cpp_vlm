import os
import io
import gc
import json
import base64
import random
import torch

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from .support.cqdm import cqdm
from .support.gguf_layers import get_layer_count
from .support.prompt_enhancer_preset import *

import folder_paths
import comfy.model_management as mm
import comfy.utils

from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    chat_handlers += ["DeepSeek-OCR"]
    _MTMD = True
except:
    _MTMD = False

chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6"]

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except:
    Gemma3ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
except:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except:
    Qwen3VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]
except:
    Qwen35ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except:
    LFM25VLChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except:
    GraniteDoclingChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
    chat_handlers += ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]
except:
    MiniCPMv45ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import MiniCPMv46ChatHandler
    chat_handlers += ["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"]
except:
    MiniCPMv46ChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
    chat_handlers += ["PaddleOCR-VL-1.5"]
except:
    PaddleOCRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    chat_handlers += ["Qwen3-ASR"]
except:
    Qwen3ASRChatHandler = None
    
try:
    from llama_cpp.llama_chat_format import Step3VLChatHandler
    chat_handlers += ["Step3-VL"]
except:
    Step3VLChatHandler = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    #states = {}
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            #cls.states.clear()
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            #cls.states.pop(f"{id}", None)
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
        
    @classmethod
    def clean(cls, all=False):
        try:
            cls.llm.close()
        except Exception:
            pass
            
        try:
            cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        if all:
            cls.clean_state()
        
        gc.collect()
        mm.soft_empty_cache()
    
    @classmethod
    def load_model(cls, config):
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking":
                    return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen3-ASR":
                    return Qwen3ASRChatHandler
                case "Qwen2.5-VL"|"MinerU2.5-Pro":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv45ChatHandler
                case "MiniCPM-v4.6"|"MiniCPM-v4.6-Thinking":
                    return MiniCPMv46ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "DeepSeek-OCR":
                    return MTMDChatHandler
                case "PaddleOCR-VL-1.5":
                    return PaddleOCRChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknow model type: "{chat_handler}"')
        
        cls.clean(all=True)
        cls.current_config = config.copy()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        n_gpu_layers = -1
        
        model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
        handler = get_chat_handler(chat_handler)
        
        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path)  * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers
        
        if mmproj and mmproj != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
            if chat_handler == "None":
                raise ValueError('"chat_handler" cannot be None!')
            
            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path)  * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))
            
            print(f"[llama-cpp_vlm] Loading clip:  {mmproj}")
            
            think_mode = "Thinking" in chat_handler
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif chat_handler in ["MiniCPM-v4.5", "GLM-4.6V", "Qwen3.5"]:
                kwargs["enable_thinking"] = think_mode

            if _MTMD:
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens

            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nPlease update llama-cpp-python from 'https://github.com/JamePeng/llama-cpp-python/releases'")

        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            if handler is not None:
                cls.chat_handler = handler(verbose=False)
            else:
                cls.chat_handler = None
        
        print(f"[llama-cpp_vlm] Loading model: {model}")
        print(f"[llama-cpp_vlm] n_gpu_layers = {n_gpu_layers}")
        cls.llm = Llama(model_path, chat_handler=cls.chat_handler, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)

any_type = AnyType("*")

if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp_vlm] Model cleanup hook applied!")

llm_extensions = ['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.gguf']
folder_paths.folder_names_and_paths["LLM"] = ([os.path.join(folder_paths.models_dir, "LLM")], llm_extensions)
preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.',
    "Qwen Tagging - 中文结果 [*]": "角色：专业 LoRA 打标工程师\n职责：根据输入图像 + 要求，生成高质量、标准化、可直接训练的 LoRA 打标标签。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **完全固定**：人物的脸型、五官、肤色、身材（所有这些特征仅绑定到唯一触发词，绝对不能出现在标签中）\n❌ **自由可变**：景别、构图、姿态、情绪、服装、配饰、道具、背景、灯光（所有这些元素必须用准确、简洁的中文完整描述）\n\n【二】不可违反的铁律\n1.  **触发词强制规则**：每张图的标签**必须以触发词 `#` 作为开头**，这是唯一的人物触发词，全程唯一，另外触发词后加上“一个女孩，单人”\n2.  **固定特征零描述规则**：所有与人物的脸型、肤色、身材的特征禁止描述**绝对禁止出现在标签的任何位置**\n3.  **可变特征全描述规则**：所有可修改的元素，必须用准确、简洁的中文完整描述，不得遗漏，包括：\n    - 景别/角度：图片的景别和角度（如正面、侧面、右前方、站姿、坐姿、仰视、俯视、特写、中景、全身等）\n    - 姿态：描述姿态、肢体动作\n    - 情绪神态：人物的表情气质（如温婉平静、清冷疏离、自然微笑等）\n    - 服装细节：衣服的款式、颜色、纹样、材质、细节\n    - 配饰：耳饰、腰饰、项链、发饰等所有饰品\n    - 道具：人物手持/身边的物品\n    - 背景：环境、场景\n    - 灯光：灯光和氛围\n4.  **标签结构统一规则**：每张图的标签必须严格按照以下顺序排列，不得打乱顺序，保证模型学习逻辑一致：\n    触发词 → 景别和构图→ 姿态 → 情绪 → 服装 → 配饰 → 道具 → 背景 → 灯光\n5.  **用词规范规则**：\n    - 全程使用**纯中文打标**，只有触发词可以为英文，不得混用其他英文。\n    - 用词精准简洁，避免冗余、重复，同一类元素用词统一（如“挂脖上衣”全程统一，不得改为“挂脖衫”等）。\n    - 不得添加任何无关标签、主观评价、冗余修饰（如“非常好看的”“精致的”等）。\n6.  **画质标签统一规则**：每张图的标签末尾，必须添加统一的画质标签：`8k高清, 写实风格, 细节拉满`，不得修改、遗漏。\n\n【三】示例参考（以此为基础标准，你生成的内容要各更多更丰富）\n示例1输出标签：\n#, 1个女孩，正面角度，脸部特写，站姿，情绪平静，露肩, 上衣是紫色挂脖高领旗袍，白色荷叶边露肩的袖套,深紫色宽腰带，腰带中央是金色花形腰扣，淡紫色披风，蓝色蝴蝶发饰，紫色花朵发饰, 蓝绿色菱形耳坠, 墨绿色纯色背景, 棚拍柔光，8k高清, 写实风格, 细节拉满\n\n示例2输出标签：\n#, 1个女孩，侧面角度，中景人像，人物坐在汽车后排座位，自然微笑，米白色圆领泡泡袖短袖上衣，下装搭配浅蓝色牛仔裤，高马尾发型，背景在汽车内，窗外是模糊的街道，暖金色阳光从车窗斜射而入，8k高清, 写实风格, 细节拉满\n\n【四】绝对禁止出现在标签中的内容\n- 除触发词外的其他英文\n- 冗余修饰、主观评价、无关标签\n- 打乱标签顺序\n- 遗漏可变特征的描述\n- 重复用词、冗余描述",
    "Qwen Tagging - 英文结果 [*]": "角色：专业 LoRA 打标工程师\n职责：根据输入图像 + 要求，生成高质量、标准化、可直接训练的 LoRA 打标标签。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **完全固定**：人物的脸型、五官、肤色、身材（所有这些特征仅绑定到唯一触发词，绝对不能出现在标签中）\n❌ **自由可变**：景别、构图、姿态、情绪、服装、配饰、道具、背景、灯光（所有这些元素必须用准确、简洁的英文完整描述）\n\n【二】不可违反的铁律\n1.  **触发词强制规则**：每张图的标签**必须以触发词 `#` 作为开头**，这是唯一的人物触发词，全程唯一，另外触发词后加上“1girl, solo”\n2.  **固定特征零描述规则**：所有与人物的脸型、肤色、身材的特征禁止描述**绝对禁止出现在标签的任何位置**\n3.  **可变特征全描述规则**：所有可修改的元素，必须用准确、简洁的英文完整描述，不得遗漏，包括：\n    - 景别/角度：图片的景别和角度（如 front view、side view、right three-quarter view、standing、sitting、low angle、close-up、medium shot、full body 等）\n    - 姿态：描述姿态、肢体动作\n    - 情绪神态：人物的表情气质（如 gentle and calm、cold and aloof、natural smile 等）\n    - 服装细节：衣服的款式、颜色、纹样、材质、细节\n    - 配饰：耳饰、腰饰、项链、发饰等所有饰品\n    - 道具：人物手持/身边的物品\n    - 背景：环境、场景\n    - 灯光：灯光和氛围\n4.  **标签结构统一规则**：每张图的标签必须严格按照以下顺序排列，不得打乱顺序，保证模型学习逻辑一致：\n    触发词 → 景别和构图→ 姿态 → 情绪 → 服装 → 配饰 → 道具 → 背景 → 灯光\n5.  **用词规范规则**：\n    - 全程使用**纯英文打标**，只有触发词可以为非英文（如中文拼音），不得混用其他语言。\n    - 用词精准简洁，避免冗余、重复，同一类元素用词统一（如 “qipao top” 全程统一，不得改为 “qipao shirt” 等）。\n    - 不得添加任何无关标签、主观评价、冗余修饰（如 \"very beautiful\"、\"exquisite\" 等）。\n6.  **画质标签统一规则**：每张图的标签末尾，必须添加统一的画质标签：`8k, realistic, highly detailed`，不得修改、遗漏。\n\n【三】示例参考（以此为基础标准，你生成的内容要各更多更丰富）\n示例1输出标签：\n#, 1girl, solo, front view, close-up of face, standing, calm expression, off-shoulder, purple high-neck qipao top, white ruffled off-shoulder sleeves, dark purple wide waist belt with golden flower buckle at the center, light purple cape, blue butterfly hair ornament, purple flower hair ornament, blue-green rhombus drop earrings, dark green plain background, studio soft lighting, 8k, realistic, highly detailed\n\n示例2输出标签：\n#, 1girl, solo, side view, medium shot, sitting in the back seat of a car, natural smile, cream round-neck puffy short-sleeve top, light blue jeans, high ponytail, car interior in background, blurry street visible through the window, warm golden sunlight streaming in from the side window, 8k, realistic, highly detailed\n\n【四】绝对禁止出现在标签中的内容\n- 除触发词外的其他非英文字符（如中文）\n- 冗余修饰、主观评价、无关标签\n- 打乱标签顺序\n- 遗漏可变特征的描述\n- 重复用词、冗余描述",
    "Qwen Describe - 详细元素描述": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的中文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】用词规范\n- 使用纯中文，用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【四】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n\n示例输出：\n正面角度，脸部特写构图，站姿，左臂自然下垂，右手轻抚发梢，情绪平静淡然，紫色挂脖高领旗袍上衣，白色荷叶边露肩袖套，深紫色宽腰带，腰带中央是金色花形腰扣，淡紫色披风垂至膝下，蓝色蝴蝶发饰与紫色花朵发饰点缀发间，蓝绿色菱形耳坠，墨绿色纯色背景，棚拍柔光，主光源从右前方45度打来，整体色调为紫色与墨绿色的低饱和对比，视觉重心集中在人物上半身",
    "Qwen Describe - 真实照片化 [*]": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的中文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n✅ **真实照片化**：如果输入图像不是真实照片（如动漫、插画、3D 渲染、游戏画面等），输出时必须将画面描述成真实照片的样子，按真实摄影的标准措辞\n✅ **必须描述人物性别**：明确说明人物是男性还是女性（如一位女孩、一位少年）\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n0. 人物性别：明确说明人物为男性或女性（如一位女孩、一位少年）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系（按真实织物的质感描述，如棉麻、丝绸、皮革的纹理与垂坠感）\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品（按真实金属/宝石的光泽与质感描述）\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系（按真实环境的材质与结构描述）\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比（真实摄影的光线表现：如实际存在的影棚灯、窗光、环境光）\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】真实照片化规则（输入非真实照片时必须遵守）\n- 一律用真实摄影的视角描述：真实材质、真实光影、镜头虚化、景深、颗粒噪点、色彩宽容度等\n- 画面中任何风格化、夸张、简化、非写实的元素，一律改写成现实世界中对应的真实形式\n- **绝对禁止**出现“动漫”“插画”“二次元”“卡通”“3D”“渲染”“游戏”“CG”“手绘”“扁平”“线条”“赛璐璐”等暴露非真实照片性质的词语\n\n【四】用词规范\n- 使用纯中文，用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【五】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n- 如果用户提供了摄影风格提示 `#`，将 `#` 内容以“按 `#` 的摄影风格拍摄”的形式附加在描述末尾\n\n示例输入（动漫截图）：\n一个站着的女孩，紫色旗袍，蓝色蝴蝶发饰，墨绿色背景\n示例输出：\n一位女孩，正面角度，脸部特写构图，站姿，左臂自然下垂，右手轻抚发梢，情绪平静淡然，紫色丝绸挂脖高领旗袍上衣，面料带有细腻的光泽和自然的垂坠感，白色棉质荷叶边露肩袖套，深紫色真皮宽腰带，腰带中央是金色黄铜花形腰扣，淡紫色纱质披风垂至膝下，蓝色金属蝴蝶发夹与紫色绢花发饰点缀发间，蓝绿色玻璃水滴形耳坠，墨绿色天鹅绒背景幕布，影棚柔光箱打光，主光源从右前方45度打来，背景虚化形成浅景深，画面略带轻微噪点，整体色调为紫色与墨绿色的低饱和对比，视觉重心集中在人物上半身",
    "Qwen Describe - 真实照片化 英文结果 [*]": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的英文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n✅ **真实照片化**：如果输入图像不是真实照片（如动漫、插画、3D 渲染、游戏画面等），输出时必须将画面描述成真实照片的样子，按真实摄影的标准措辞\n✅ **必须描述人物性别**：明确说明人物是男性还是女性（如一位女孩、一位少年）\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n0. 人物性别：明确说明人物为男性或女性（如一位女孩、一位少年）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系（按真实织物的质感描述，如棉麻、丝绸、皮革的纹理与垂坠感）\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品（按真实金属/宝石的光泽与质感描述）\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系（按真实环境的材质与结构描述）\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比（真实摄影的光线表现：如实际存在的影棚灯、窗光、环境光）\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】真实照片化规则（输入非真实照片时必须遵守）\n- 一律用真实摄影的视角描述：真实材质、真实光影、镜头虚化、景深、颗粒噪点、色彩宽容度等\n- 画面中任何风格化、夸张、简化、非写实的元素，一律改写成现实世界中对应的真实形式\n- **绝对禁止**出现“动漫”“插画”“二次元”“卡通”“3D”“渲染”“游戏”“CG”“手绘”“扁平”“线条”“赛璐璐”等暴露非真实照片性质的词语\n\n【四】用词规范\n- 输出使用**纯英文**，只有触发词等专有名词（如 qipao、kimono）可使用中文拼音，其余一律用英文\n- 用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【五】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n- 如果用户提供了摄影风格提示 `#`，将 `#` 内容以“shot in the style of `#`”的形式附加在描述末尾\n\nExample input (anime screenshot):\nA standing girl, purple qipao, blue butterfly hair ornament, dark green background\nExample output:\nA girl, front view, close-up composition, standing posture, left arm hanging naturally, right hand lightly touching her hair, calm and composed expression, purple silk high-neck qipao top with delicate sheen and natural drape, white cotton ruffled off-shoulder sleeves, dark purple genuine leather waist belt with a golden brass flower buckle at the center, light purple gauze cape falling to the knees, blue metal butterfly hair clip and purple silk flower hair ornament, blue-green glass teardrop earrings, dark green velvet backdrop, studio softbox lighting, key light from 45 degrees front-right, shallow depth of field with blurred background, slight film grain, overall low-saturation contrast of purple and dark green, visual focus on the upper body",
    "Qwen Describe - 动漫风格化 英文结果 [*]": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的英文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n✅ **动漫风格化**：如果输入图像是真实照片，输出时必须将画面描述成动漫插画的样子，按动画原画的风格措辞\n✅ **必须描述人物性别**：明确说明人物是男性还是女性（如一位女孩、一位少年）\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n0. 人物性别：明确说明人物为男性或女性（如一位女孩、一位少年）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系（按动画上色的质感描述，如色块、高光、褶皱线条）\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品（按插画风格的光泽与造型描述）\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系（按插画背景的风格与色调描述）\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比（按动画特有的光影表现描述：如大块高光、渐变阴影、氛围光）\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】动漫风格化规则（输入真实照片时必须遵守）\n- 一律用动画插画的视角描述：赛璐璐上色、干净的线条、大块平涂色块、夸张的高光与阴影、无噪点、画面干净\n- 画面中所有写实、复杂的细节，一律改写成动漫插画中对应的简化、风格化形式\n- **绝对禁止**出现“照片”“摄影”“相机”“胶片”“镜头”“噪点”“颗粒”“景深”“真实感”“写实”等暴露真实照片性质的词语\n\n【四】用词规范\n- 输出使用**纯英文**，只有触发词等专有名词（如 qipao、kimono）可使用中文拼音，其余一律用英文\n- 用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【五】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n- 如果用户提供了动漫风格提示 `#`，将 `#` 内容以“drawn in the style of `#`”的形式附加在描述末尾\n\nExample input (real photo):\nA standing girl, purple qipao, blue butterfly hair ornament, dark green background\nExample output:\nA girl, front view, close-up composition, standing posture, left arm hanging naturally, right hand lightly touching her hair, calm and composed expression, purple high-neck qipao top with large flat purple color blocks and pleat fold lines, fine highlight outlines on the fabric edges, white ruffled off-shoulder sleeves, dark purple wide waist belt with a golden flower buckle at the center, light purple cape falling to the knees, blue butterfly hair ornament and purple flower hair ornament, blue-green rhombus drop earrings, dark green plain background, studio soft lighting, key light from 45 degrees front-right, bright anime-style highlights on the hair strands and clothing corners, clean cel-shaded image without noise, overall low-saturation contrast of purple and dark green, visual focus on the upper body",
    "Qwen Describe - 黑白漫画彩色真实化 [*]": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的中文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n✅ **彩色真实化**：如果输入图像是黑白漫画（黑白线稿、页漫、条漫等），输出时必须将画面描述成**彩色的真实照片**的样子，按真实摄影的标准措辞，并为画面元素合理地补全颜色\n✅ **必须描述人物性别**：明确说明人物是男性还是女性（如一位女孩、一位少年）\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n0. 人物性别：明确说明人物为男性或女性（如一位女孩、一位少年）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系（按真实织物的质感描述，如棉麻、丝绸、皮革的纹理与垂坠感）\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品（按真实金属/宝石的光泽与质感描述）\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系（按真实环境的材质与结构描述）\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比（真实摄影的光线表现：如实际存在的影棚灯、窗光、环境光）\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】彩色真实化规则（输入黑白漫画时必须遵守）\n- 一律用真实摄影的视角描述：真实材质、真实光影、镜头虚化、景深、颗粒噪点、色彩宽容度等\n- 黑白画面没有颜色信息，必须为服装、配饰、道具、背景、灯光等元素**合理推测并补全颜色**，颜色要自然协调、符合画面内容\n- 画面中任何风格化、夸张、简化、非写实的元素，一律改写成现实世界中对应的真实形式\n- **绝对禁止**出现“黑白”“漫画”“线稿”“素描”“插画”“二次元”“卡通”“3D”“渲染”“赛璐璐”等暴露输入性质的词语\n\n【四】用词规范\n- 使用纯中文，用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【五】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n- 如果用户提供了摄影风格提示 `#`，将 `#` 内容以“按 `#` 的摄影风格拍摄”的形式附加在描述末尾\n\n示例输入（黑白漫画）：\n一个站着的女孩，穿旗袍，戴蝴蝶发饰，背景是纯色\n示例输出：\n一位女孩，正面角度，脸部特写构图，站姿，左臂自然下垂，右手轻抚发梢，情绪平静淡然，紫色丝绸挂脖高领旗袍上衣，面料带有细腻的光泽和自然的垂坠感，白色棉质荷叶边露肩袖套，深紫色真皮宽腰带，腰带中央是金色黄铜花形腰扣，淡紫色纱质披风垂至膝下，蓝色金属蝴蝶发夹与紫色绢花发饰点缀发间，蓝绿色玻璃水滴形耳坠，墨绿色天鹅绒背景幕布，影棚柔光箱打光，主光源从右前方45度打来，背景虚化形成浅景深，画面略带轻微噪点，整体色调为紫色与墨绿色的低饱和对比，视觉重心集中在人物上半身",
    "Qwen Describe - 黑白漫画彩色真实化 英文结果 [*]": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的英文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n✅ **彩色真实化**：如果输入图像是黑白漫画（黑白线稿、页漫、条漫等），输出时必须将画面描述成**彩色的真实照片**的样子，按真实摄影的标准措辞，并为画面元素合理地补全颜色\n✅ **必须描述人物性别**：明确说明人物是男性还是女性（如一位女孩、一位少年）\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n0. 人物性别：明确说明人物为男性或女性（如一位女孩、一位少年）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系（按真实织物的质感描述，如棉麻、丝绸、皮革的纹理与垂坠感）\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品（按真实金属/宝石的光泽与质感描述）\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系（按真实环境的材质与结构描述）\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比（真实摄影的光线表现：如实际存在的影棚灯、窗光、环境光）\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】彩色真实化规则（输入黑白漫画时必须遵守）\n- 一律用真实摄影的视角描述：真实材质、真实光影、镜头虚化、景深、颗粒噪点、色彩宽容度等\n- 黑白画面没有颜色信息，必须为服装、配饰、道具、背景、灯光等元素**合理推测并补全颜色**，颜色要自然协调、符合画面内容\n- 画面中任何风格化、夸张、简化、非写实的元素，一律改写成现实世界中对应的真实形式\n- **绝对禁止**出现“黑白”“漫画”“线稿”“素描”“插画”“二次元”“卡通”“3D”“渲染”“赛璐璐”等暴露输入性质的词语\n\n【四】用词规范\n- 输出使用**纯英文**，只有触发词等专有名词（如 qipao、kimono）可使用中文拼音，其余一律用英文\n- 用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【五】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n- 如果用户提供了摄影风格提示 `#`，将 `#` 内容以“shot in the style of `#`”的形式附加在描述末尾\n\nExample input (black-and-white manga):\nA standing girl, wearing a qipao, butterfly hair ornament, plain background\nExample output:\nA girl, front view, close-up composition, standing posture, left arm hanging naturally, right hand lightly touching her hair, calm and composed expression, purple silk high-neck qipao top with delicate sheen and natural drape, white cotton ruffled off-shoulder sleeves, dark purple genuine leather waist belt with a golden brass flower buckle at the center, light purple gauze cape falling to the knees, blue metal butterfly hair clip and purple silk flower hair ornament, blue-green glass teardrop earrings, dark green velvet backdrop, studio softbox lighting, key light from 45 degrees front-right, shallow depth of field with blurred background, slight film grain, overall low-saturation contrast of purple and dark green, visual focus on the upper body",
    "Qwen Describe - 动漫风格化 [*]": "角色：专业图像描述工程师\n职责：详细、完整地描述画面中的各个元素，输出可直接用于图像生成参考的中文描述。\n**必须100%严格遵守以下所有规则，不得违反任何一条**：\n【一】核心目标\n✅ **必须详细描述**：画面中的所有可变元素，逐项展开、不遗漏、有层次\n✅ **动漫风格化**：如果输入图像是真实照片，输出时必须将画面描述成动漫插画的样子，按动画原画的风格措辞\n✅ **必须描述人物性别**：明确说明人物是男性还是女性（如一位女孩、一位少年）\n❌ **绝对禁止**：人物的长相（脸型、五官、肤色）与身材（身高、胖瘦、体型）——这些特征不得出现在描述中的任何位置\n\n【二】必须详细描述的元素（按以下顺序逐项展开）\n0. 人物性别：明确说明人物为男性或女性（如一位女孩、一位少年）\n1. 景别/角度：正面、侧面、右前方、仰视、俯视、特写、中景、全身等\n2. 姿态动作：人物的姿势、肢体动作、动态\n3. 情绪神态：表情气质（如温婉平静、清冷疏离、自然微笑等）\n4. 服装细节：款式、颜色、纹样、材质、层次与搭配关系（按动画上色的质感描述，如色块、高光、褶皱线条）\n5. 配饰：耳饰、发饰、项链、腰饰、鞋履等所有饰品（按插画风格的光泽与造型描述）\n6. 道具：人物手持或身边的物品及其细节\n7. 背景环境：场景、空间、远近层次、前景中景背景关系（按插画背景的风格与色调描述）\n8. 灯光氛围：光源方向、色温、光线质感、明暗对比（按动画特有的光影表现描述：如大块高光、渐变阴影、氛围光）\n9. 构图色彩：画面构图方式、主色调、色彩搭配与视觉重点\n\n【三】动漫风格化规则（输入真实照片时必须遵守）\n- 一律用动画插画的视角描述：赛璐璐上色、干净的线条、大块平涂色块、夸张的高光与阴影、无噪点、画面干净\n- 画面中所有写实、复杂的细节，一律改写成动漫插画中对应的简化、风格化形式\n- **绝对禁止**出现“照片”“摄影”“相机”“胶片”“镜头”“噪点”“颗粒”“景深”“真实感”“写实”等暴露真实照片性质的词语\n\n【四】用词规范\n- 使用纯中文，用词精准简洁，避免冗余与重复\n- 只做客观描述，不做主观评价（如“非常好看”“精致”“梦幻”等）\n\n【五】输出格式\n- 以自然语言段落形式输出，按【二】中元素顺序组织，详细而清晰\n- 如果用户提供了动漫风格提示 `#`，将 `#` 内容以“采用 `#` 的作画风格”的形式附加在描述末尾\n\n示例输入（真实照片）：\n一个站着的女孩，紫色旗袍，蓝色蝴蝶发饰，墨绿色背景\n示例输出：\n一位女孩，正面角度，脸部特写构图，站姿，左臂自然下垂，右手轻抚发梢，情绪平静淡然，紫色挂脖高领旗袍上衣，大面积平涂紫色色块配合裙摆褶皱线条，衣料边缘带细腻的高光描边，白色荷叶边露肩袖套，深紫色宽腰带，腰带中央是金色花形腰扣，淡紫色披风垂至膝下，蓝色蝴蝶发饰与紫色花朵发饰点缀发间，蓝绿色菱形耳坠，墨绿色纯色背景，棚拍柔光，主光源从右前方45度打来，发丝和衣角带有明亮的动画式高光，画面干净无噪点，整体色调为紫色与墨绿色的低饱和对比，视觉重心集中在人物上半身",
}
preset_tags = list(preset_prompts.keys())

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def parse_json(json_str):
    json_output = json_str.strip().removeprefix("```json").removesuffix("```")
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    resized_frames = []
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(img_resized)

def qwen3bbox(image, json):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    for item in json:
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    
    for item in json:
        try:
            label = item["label"]
        except Exception:
            try:
                label = item["text_content"]
            except Exception:
                label = "bbox"
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)
        
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        model_list = [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"]+[f for f in all_llms if "mmproj" in f.lower()]
            
        return {"required": {
            "model": (model_list, {"tooltip": "LLM 目录下的 GGUF 语言模型文件（自动排除 mmproj 视觉模块）。"}),
            "mmproj": (mmproj_list, {
                "default": "None",
                "tooltip": "视觉编码器（mmproj）文件。图像/视频输入时必须选择，且需与模型匹配。"
            }),
            "chat_handler": (chat_handlers, {
                "default": "None",
                "tooltip": "对话模板处理器，必须与模型的架构匹配（如 Qwen3.5 选 Qwen3.5，LLaVA 架构选 LLaVA-1.5）。\n图像输入时不能为 None。"
            }),
            "n_ctx": ("INT", {
                "default": 8192,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "上下文长度限制。越小占用显存越少；模型训练上下文较大时无需追求满载。"
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "显存使用上限（GB），-1 = 不限（全部层进 GPU）。\n设置后会自动计算可放入 GPU 的层数，剩余层跑 CPU。实际占用可能略超。"
            }),
            "image_min_tokens": ("INT", {
                "default": 0, "min": 0, "max": 4096, "step": 32,
                "tooltip": "图像最小 token 数（Qwen3-VL / Qwen3.5 系列）。0 = 使用模型默认值。"
            }),
            "image_max_tokens": ("INT", {
                "default": 0, "min": 0, "max": 4096, "step": 32,
                "tooltip": "图像最大 token 数（Qwen3-VL / Qwen3.5 系列）。0 = 使用模型默认值。\n调小可节省显存与显存占用，但会损失图像细节。"
            }),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    '''
    @classmethod
    def IS_CHANGED(s, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        config_str = json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
        return config_str
    '''
    def loadmodel(self, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            print("[llama-cpp_vlm] Loading model...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)

class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {
                    "default": preset_tags[1],
                    "tooltip": "内置提示词预设。\n"
                               "@ 代表图像（视频模式自动替换为 video），# 为自定义内容占位符。\n"
                               "带 * 的预设会使用 custom_prompt 填充占位符；其余预设只要填写了 custom_prompt 就会被完全替换。\n\n"
                               "Empty - Nothing: 不使用预设\n"
                               "Normal - Describe: 简单描述图片\n"
                               "Prompt Style - Tags: 生成最多 50 个逗号分隔的视觉标签（禁抽象概念）\n"
                               "Prompt Style - Simple: 强制单句简洁描述\n"
                               "Prompt Style - Detailed: 2-3 句详细描述\n"
                               "Prompt Style - Extreme Detailed: 极详细长段落描述\n"
                               "Prompt Style - Cinematic: 电影风格化提示词（主体/姿势/环境/灯光/风格）\n"
                               "Creative - Detailed Analysis: 按主体/服饰/背景等分节分析\n"
                               "Creative - Summarize Video: 总结视频关键事件（视频模式）\n"
                               "Creative - Short Story: 看图创作短故事\n"
                               "Creative - Refine & Expand Prompt: 扩写增强已有提示词\n"
                               "Vision - *Bounding Box: 输出 bbox 定位 JSON（# 填目标类别）\n"
                               "Qwen Tagging - 中文结果 [*]: LoRA 打标规则书（中文输出），# 填触发词\n"
                               "Qwen Tagging - 英文结果 [*]: LoRA 打标规则书（英文输出），# 填触发词\n"
                               "Qwen Describe - 详细元素描述: 中文段落式详细描述画面元素（禁描述长相与身材）\n"
                               "Qwen Describe - 真实照片化 [*]: 同详细元素描述，非真实照片时按真实摄影标准输出（# 可填风格提示）\n"
                               "Qwen Describe - 动漫风格化 [*]: 同详细元素描述，真实照片时按动画插画风格输出（# 可填作画风格）\n"
                               "Qwen Describe - 真实照片化 英文结果 [*]: 同上，输出为英文\n"
                               "Qwen Describe - 动漫风格化 英文结果 [*]: 同上，输出为英文\n"
                               "Qwen Describe - 黑白漫画彩色真实化 [*]: 黑白漫画按彩色真实照片描述，并合理补全颜色\n"
                               "Qwen Describe - 黑白漫画彩色真实化 英文结果 [*]: 同上，输出为英文",
                }),
                "custom_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.',
                    "tooltip": "自定义提示词。\n若预设名带 *（如 Bounding Box），此处内容填入预设的 # 占位符；\n否则只要非空，就会完全覆盖预设提示词。"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "系统提示词，位于对话最前，用于设定角色/全局规则。\n修改后会自动清空对话历史。"
                }),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: 逐张读取图像（每张独立生成）\nimages: 一次读取全部图像（多图共答）\nvideo: 将输入图像序列当作视频处理"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "从输入视频中均匀采样的帧数（仅 video 模式生效）。"
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": '图像缩放的最大边长（images 和 video 模式）。调小可加快处理、节省显存。'
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                    "tooltip": "随机种子。固定后同图同参数输出可复现。"
                }),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "推理完成后立即卸载模型，释放显存（下次执行会重新加载）。"
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "在内存中保存本轮对话历史，多轮对话可延续上下文。\n（会占用显存/内存，不需要时建议关闭）"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("LLAMACPPARAMS", {
                    "tooltip": "采样参数（来自 Llama-cpp Parameters 节点）。不连接则使用默认采样参数。"
                }),
                "images": ("IMAGE", {
                    "tooltip": "图像/视频帧输入（支持多图或批处理）。"
                }),
                "queue_handler": (any_type, {
                    "tooltip": "用于控制多个 Instruct 节点的执行顺序（来自同类的 queue_handler 输出）。"
                }),
            },
            
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("output", "output_list", "state_uid", "final_prompt")
    OUTPUT_IS_LIST = (False, True, False, False)
    OUTPUT_TOOLTIPS = ("模型生成的最终文本（单条）", "模型生成的最终文本（列表，多图逐条）", "对话状态 ID", "本次实际传给模型的最终用户提示词（预设替换后）")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def sanitize_messages(self, messages):
        clean_messages = messages.copy()
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, images=None, queue_handler=None):
        if not LLAMA_CPP_STORAGE.llm:
            LLAMA_CPP_STORAGE.load_model(llama_model)
            #raise RuntimeError("The model has been unloaded or failed to load!")
        
        if parameters is None:
            parameters = {}
        
        if _MTMD:
            parameters.pop("present_penalty", None)
            
        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
        
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        if last_sys_prompt != system_prompts:
            messages = []
            LLAMA_CPP_STORAGE.clean_state()
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            if save_states:
                try:
                    print(f"[llama-cpp_vlm] Loading state and history id={uid}...")
                    #LLAMA_CPP_STORAGE.llm.load_state(LLAMA_CPP_STORAGE.states[f"{uid}"])
                    messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", [])
                except Exception as e:
                    messages = []
            else:
                messages = []
        out1 = ""
        out2 = []
        user_content = []
        prompt_text = ""
        if custom_prompt.strip() and "*" not in preset_prompt:
            prompt_text = custom_prompt
            user_content.append({"type": "text", "text": custom_prompt})
        else:
            p = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            prompt_text = p
            user_content.append({"type": "text", "text": p})
            
        if images is not None:
            if not hasattr(LLAMA_CPP_STORAGE.chat_handler, "clip_model_path") or LLAMA_CPP_STORAGE.chat_handler.clip_model_path is None:
                 raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")
                
            frames = images
            if video_input:
                indices = np.linspace(0, len(images) - 1, max_frames, dtype=int)
                frames = [images[i] for i in indices]
                
            if inference_mode == "one by one":
                tmp_list = []
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": ""}
                }
                user_content.append(image_content)
                messages.append({"role": "user", "content": user_content})
                print(f"[llama-cpp_vlm] Start processing {len(frames)} images")
                
                for i, image in enumerate(cqdm(frames)):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    for item in user_content:
                        if item.get("type") == "image_url":
                            item["image_url"]["url"] = f"data:image/jpeg;base64,{data}"
                            break
                    output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
                    text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    out2.append(text)
                    if len(frames) > 1:
                        tmp_list.append(f"====== Image {i+1} ======")
                    tmp_list.append(text)
                    
                out1 = "\n\n".join(tmp_list)
            else:
                for image in frames:
                    if len(frames) > 1:
                        data = image2base64(scale_image(image, max_size))
                    else:
                        data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"}
                    }
                    user_content.append(image_content)
                    
                messages.append({"role": "user", "content": user_content})
                output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
                out1 = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                out2 = [out1]
        else:
            messages.append({"role": "user", "content": user_content})
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
            out1 = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            out2 = [out1]
            
        if save_states:
            print(f"[llama-cpp_vlm] Saving state id={uid}...")
            #LLAMA_CPP_STORAGE.states[f"{uid}"] = LLAMA_CPP_STORAGE.llm.save_state()
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            if not LLAMA_CPP_STORAGE.messages.get(f"{uid}"):
                LLAMA_CPP_STORAGE.sys_prompts.pop(f"{uid}", None)
                
        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            if LLAMA_CPP_STORAGE.current_config["chat_handler"] in ["Qwen3.5", "Qwen3.5-Thinking"]:
                LLAMA_CPP_STORAGE.llm.n_tokens = 0
                LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                if LLAMA_CPP_STORAGE.llm.is_hybrid and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                    LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
            
        del messages
        gc.collect()
        return (out1, out2, uid, prompt_text)

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "单次生成的最大 token 数。0 = 不限（直到结束符或上下文用尽）。"
                }),
                "top_k": ("INT", {
                    "default": 30, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "仅从概率最高的前 K 个 token 中采样。0 = 禁用。值越小输出越保守。"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "核采样：累计概率达到该值的最小 token 集合内采样。1.0 = 禁用。"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "最小概率阈值：概率低于（最高概率 × min_p）的 token 被剔除。0 = 禁用。"
                }),
                "typical_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "典型采样参数。1.0 = 禁用。"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "采样温度：越高输出越发散/有创意，越低越确定/保守。\n建议 0.7-1.0；出现复读可调高。"
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "重复惩罚：>1 抑制近期重复 token（作用于最近窗口）。\n模型复读时建议 1.1-1.3。"
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "频率惩罚：对全序列中出现过的 token 按出现次数惩罚，抑制整体重复。"
                }),
                "present_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "存在惩罚：对已出现的 token 给予固定惩罚，鼓励话题切换（不区分次数）。"
                }),
                #"tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                #"penalty_last_n": ("INT", {"default": 64, "min": -1, "max": 8192, "step": 1}),
                "mirostat_mode": ("INT", {
                    "default": 0, "min": 0, "max": 2, "step": 1,
                    "tooltip": "Mirostat 自适应采样模式：0 = 禁用，1 = Mirostat，2 = Mirostat 2.0。"
                }),
                "mirostat_eta": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mirostat 学习率：越低对文本的适应越慢。"
                }),
                "mirostat_tau": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Mirostat 目标困惑度（perplexity）。"
                }),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "对话状态的保存 ID。-1 = 使用节点的 unique_id。\n配合 Instruct 节点的 save_states 使用。"
                }),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    def process(self, **kwargs):
        return (kwargs,)
    
class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Clear the saved state for a specific ID (-1 = clear all)"
                }),
            },
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any, state_uid):
        print(f"[llama-cpp_vlm] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        print("[llama-cpp_vlm] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels."
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode = mode[0]
        label = label[0]

        flat_images_list = []
        original_structure = []
    
        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            
            if label != "":
                try:
                    bboxes = [item for item in bboxes if item["label"] == label]
                except Exception:
                    bboxes = [item for item in bboxes if item.get("text_content") == label]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]
                        
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)
                    
            if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes]
                
            output_bboxes.append(bbox)
            
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
            
        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper
        
    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")
    
class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        
        seg_list = []
        image_for_cropping = image[0] 
        
        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            
            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                print(f"Warning: Skipping bbox with invalid expanded size: {crop_region}")
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            cropped_mask_np = local_mask_np
            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            
            src_x_start = max(0, x1_exp)
            src_y_start = max(0, y1_exp)
            src_x_end = min(width, x2_exp)
            src_y_end = min(height, y2_exp)
            
            dst_x_start = src_x_start - x1_exp
            dst_y_start = src_y_start - y1_exp
            dst_x_end = src_x_end - x1_exp
            dst_y_end = src_y_end - y1_exp
            
            if src_x_end > src_x_start and src_y_end > src_y_start:
                source_crop = image_for_cropping[src_y_start:src_y_end, src_x_start:src_x_end, :]
                cropped_img_padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = source_crop
                
            cropped_image_tensor = cropped_img_padded.permute(2, 0, 1).unsqueeze(0)
            
            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            
            seg_list.append(seg)
            
        segs = (mask_shape, seg_list)
        
        return (segs,)
    
class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image, dilation, feather):
        masks = []
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        
        for i, bbox in enumerate(bboxes):
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp
            
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = dilation
            local_y1 = dilation
            local_x2 = local_x1 + (x2 - x1)
            local_y2 = local_y1 + (y2 - y1)
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0
            
            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)
                
            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)
            
            if x2_c > x1_c and y2_c > y1_c:
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = 1.0
                
            if feather > 0:
                current_full_mask_np = gaussian_filter(current_full_mask_np, sigma=feather)
                
            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)
            
        masks.append(combined_full_mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes."
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, bboxes, image_index, bbox_index):
        if bbox_index != 999:
            return ([bboxes[image_index][bbox_index]],)
        return (bboxes[image_index],)

# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING",),
                "default": ("STRING",),
            },
        }
    
    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, key=None, default=None):
        if isinstance(input, str):
            input = [input]
            
        result = {}
        for i, json in enumerate(input):
            val = ""
            if key is not None and key != "":
                val = get_nested_value(json.strip().removeprefix("```json").removesuffix("```"), key, default)
            else:
                raise ValueError("Key cannot be empty!")
            
            result["any"][i] = val
            try:
                result["string"][i] = str(val)
            except Exception as e:
                result["string"][i] = val
            
            try:
                result["int"][i] = int(val)
            except Exception as e:
                result["int"][i] = val
            
            try:
                result["float"][i] = float(val)
            except Exception as e:
                result["float"][i] = val
            
            try:
                result["boolean"][i] = val.lower() == "true"
            except Exception as e:
                result["boolean"][i] = val
                
        if len(result["any"]) == 1:
            result["any"] = result["any"][0]
            result["string"] = result["string"][0]
            result["int"] = result["int"][0]
            result["float"] = result["float"][0]
            result["boolean"] = result["boolean"][0]
        
        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
                data = json.loads(data)
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, input, label):
        if isinstance(input, str):
            input = [input]
        
        output = []
        for value in input:
            output.append(value.strip().removeprefix(f"```{label}").removesuffix("```"))
        if len(output) == 1:
            return (output[0],)
        return (output,)

class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vlm"
    
    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
                raise ValueError(f'Unknow preset: "{preset}"')
        
NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "parse_json_node": parse_json_node,
    "json_to_bbox": json_to_bbox,
    "bbox_to_segs": bbox_to_segs,
    "bbox_to_mask": bbox_to_mask,
    "bboxes_to_bbox": bboxes_to_bbox,
    "remove_code_block": remove_code_block,
    "PromptEnhancerPreset": PromptEnhancerPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Instruct",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "parse_json_node": "Parse JSON",
    "json_to_bbox": "JSON to BBoxes",
    "bbox_to_segs": "BBoxes to SEGS",
    "bbox_to_mask": "BBoxes to MASK",
    "bboxes_to_bbox": "BBoxes to BBox",
    "remove_code_block": "Unpack Code Block",
    "PromptEnhancerPreset": "Prompt Enhancer Preset",
}