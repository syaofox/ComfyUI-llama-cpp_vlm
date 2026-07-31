# AGENTS.md

## 项目概述

ComfyUI 自定义节点:基于 llama.cpp(通过 [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) fork)在 ComfyUI 内原生运行 LLM/VLM(GGUF + mmproj)。用于反推图/打标、图像描述、视频理解、bbox 检测等。

## 项目结构

- `nodes.py` — 全部节点逻辑(约 1200 行,单文件)
- `support/cqdm.py` — 进度条
- `support/gguf_layers.py` — 读取 GGUF 层数(`get_layer_count`)
- `support/prompt_enhancer_preset.py` — 各绘图模型官方 system prompt 预设
- `__init__.py` — 导出 `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`

## 运行环境(重要)

- 本地目录 `/home/syaofox/Projects/ComfyUI/custom_nodes/ComfyUI-llama-cpp_vlm` 是**源码副本**
- 实际运行:容器 `comfyui-docker`,`/mnt/github/comfyui-docker` 挂载为容器内 `/home/comfy/app`
- 模型目录:容器 `/home/comfy/app/models/LLM/`(宿主 `/mnt/github/comfyui-docker/models/LLM/`)
- GPU:RTX 3060 12GB;容器内存 16GB(可用约 6GB)
- 容器内 `pip show llama-cpp-python` = 0.3.33(JamePeng fork,支持 Qwen3.5 等新架构)
- 修改 `nodes.py` 后需同步到 `/mnt/github/comfyui-docker/custom_nodes/ComfyUI-llama-cpp_vlm/nodes.py` 并重启 ComfyUI 才生效

## 架构要点

- `LLAMA_CPP_STORAGE`(nodes.py:119)是全局单例,持有 `llm` / `chat_handler` / `messages` / `sys_prompts`,负责模型加载与卸载
- `chat_handlers` 列表通过 **try/except 探测导入**(nodes.py:26-113):不同 llama-cpp-python 版本只显示可用的 handler,缺包时对应项不出现
- `load_model()`(nodes.py:160):
  - `chat_handler == "None"` 且配置了 mmproj 时抛错(图像输入必须选 handler)
  - `vram_limit == -1` → `n_gpu_layers = -1`(全部层进 GPU);否则按 `(vram_limit - mmproj_size) / gguf_layer_size` 计算层数
  - handler 构造失败时抛 `RuntimeError` 提示更新 llama-cpp-python
- Instruct 节点 prompt 逻辑(nodes.py:539):`custom_prompt` 非空且预设名不含 `*` → 完全用 custom_prompt;否则用预设并替换占位符(`@` = image/video,`#` = custom_prompt)
- 输出统一 `(out1, out2, uid)`:`out1` 单条文本,`out2` 列表,`uid` 为 state 标识
- `image2base64` 将图像转 JPEG base64 后经 `create_chat_completion` 传给模型
- 节点注册映射在文件末尾 `NODE_CLASS_MAPPINGS`(nodes.py:1215)

## 开发约定

- **UI tooltip 一律使用中文**(用户要求),多行用 `\n` 拼接,combo 类型同样支持字符串 tooltip
- 不添加代码注释(除非用户明确要求)
- 保持单文件风格、与现有 `nodes.py` 风格一致;新节点需同时注册进两个 MAPPINGS
- 修改后必须验证:
  ```bash
  python -c "import ast; ast.parse(open('nodes.py').read())"
  ```
  并同步到 `/mnt/github/comfyui-docker/custom_nodes/ComfyUI-llama-cpp_vlm/nodes.py`
- 模型文件命名约定:主模型放 `LLM/` 下(按目录分),mmproj 文件名含 "mmproj";加载时路径填相对 `models/LLM` 的路径(如 `GGUF/xxx.gguf`)

## 常见陷阱

- **显存 OOM**:12GB 卡 + 27B 模型全层 GPU 必挂(`Failed to create context with model`)。27B 只能上 IQ2_XXS(~8.6GB)级别;打标/反推图建议 8-9B 模型 Q4_K_M(如 Qwen3.5-9B ≈ 5.7GB + mmproj ≈ 0.9GB)
- **mmproj 必须与模型配对**,不同尺寸模型(如 27B/9B)的 mmproj 不可混用
- **JoyCaption 类模型是 captioning 模型不是指令遵循模型**:长规则提示词会回显/输出废话;打标任务优先用 Qwen3.5/Qwen3-VL 系(chat_handler 选 `Qwen3.5`/`Qwen3-VL`)
- **复读问题**:parameters 节点推荐 temperature=1.0、top_k=10、top_p=0.5、repeat_penalty=1.15(JoyCaption 官方风格)
- 容器内直接 `python -c "import llama_cpp"` 会因缺 `libcudart` 报错,需 `LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib`
- 修改节点后 ComfyUI 需重启(或刷新节点列表)才加载新 tooltip/参数
