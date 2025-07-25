# ComfyUI DeZoomer Nodes

A collection of custom nodes for ComfyUI. Currently includes:

1. [**Video Captioning Node**](#video-captioning-node)
2. [**Caption Refinement Node**](#caption-refinement-node)

---

## Installation

### Option 1 (Recommended): Install via ComfyUI-Manager

1. Install [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager)
2. Open ComfyUI
3. Click on "Manager" tab
4. Click on "Custom Nodes Manager"
5. Search for "DeZoomer"
6. Click "Install" on "ComfyUI-DeZoomer-Nodes"

### Option 2: Manual Installation

<details>
<summary>Click to expand</summary>

1. Clone this repository into your ComfyUI's `custom_nodes` folder:

```bash
git clone https://github.com/De-Zoomer/ComfyUI-DeZoomer-Nodes.git
```

2. Install the required dependencies:

```bash
cd ComfyUI-DeZoomer-Nodes
pip install -r requirements.txt
```

3. If you're using the portable version of ComfyUI, run this command in the ComfyUI_windows_portable folder:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-DeZoomer-Nodes\requirements.txt
```

</details>

---

## Changelog

All notable changes to this project will be documented in this section.

<details>
<summary>Click to expand</summary>

### 1.0.3

- Add ShotVL as an option on Video Captioning node.

### 1.0.2

- Add SkyCaptioner-V1 as an option on Video Captioning node.

### 1.0.1

- Memory management improvements.

### 1.0.0

- Initial release with Video Captioning and Caption Refinement nodes.
</details>

---

### Video Captioning Node

Takes video frames and generates detailed captions using the Qwen2.5-VL model.

![Video Captioning Node](docs/images/video_captioning.png)
[Example Workflow](example_workflows/video_captioning.json)

#### Parameters

- **Images**: Input video frames to process (ComfyUI's IMAGE type)
- **User Prompt**: Detailed instructions for what aspects to analyze (default provided)
- **System Prompt**: Instructions for the model's behavior and output style
- **Model Name**: Qwen2.5-VL model to use. Also supports SkyCaptioner-V1 and ShotVL (default: "Qwen/Qwen2.5-VL-7B-Instruct")
- **Temperature**: Controls randomness in generation (default: 0.3)
- **Use Flash Attention**: Enables faster attention implementation (default: True)
- **Low CPU Memory Usage**: Optimizes for low CPU memory usage (default: True)
- **Quantization Type**: Memory optimization (4-bit or 8-bit)
- **Keep Model Loaded**: Option to keep model in memory after processing (default: False)
- **Seed**: Random seed for reproducible generation

It's a porting node from [@cseti007](https://github.com/cseti007) on his [Qwen2.5-VL-Video-Captioning](https://github.com/cseti007/Qwen2.5-VL-Video-Captioning).
The node processes video frames and generates comprehensive descriptions covering:

- Main content and characters
- Object and character details
- Actions and movements
- Background elements
- Visual style
- Camera work
- Scene transitions

#### Requirements

- CUDA-compatible GPU (recommended)
- At least 16GB of GPU memory for optimal performance

---

### Caption Refinement Node

Takes a caption and refines it using the Qwen2.5 model.

![Caption Refinement Node](docs/images/caption_refinement.png)
[Example Workflow](example_workflows/caption_refinement.json)

#### Parameters

- **Caption**: Input caption to refine (required)
- **System Prompt**: Instructions for the model's behavior and output style
- **Model Name**: Qwen2.5 model to use (default: "Qwen/Qwen2.5-7B-Instruct")
- **Temperature**: Controls randomness in generation (default: 0.7)
- **Max Tokens**: Maximum tokens for refinement output (default: 200)
- **Quantization Type**: Memory optimization (4-bit or 8-bit)
- **Keep Model Loaded**: Option to keep model in memory after processing (default: False)
- **Seed**: Random seed for reproducible generation

The node refines captions by:

- Making the text more continuous and coherent
- Removing video-specific references
- Adding clothing details
- Using only declarative sentences

#### Requirements

- CUDA-compatible GPU (recommended)
- At least 16GB of GPU memory on 4-bit quantization (>18GB GPU for 8-bit)

---

## License

This project is licensed under the GPL License - see the LICENSE file for details.

---

## Acknowledgments

This project uses the following models developed by Alibaba Cloud:

- [Qwen2.5-VL model](https://github.com/QwenLM/Qwen2.5-VL) for video captioning
- [ShotVL-7B model](https://huggingface.co/Vchitect/ShotVL-7B) for video captioning
- [SkyCaptioner-V1 model](https://huggingface.co/Skywork/SkyCaptioner-V1) for video captioning
- [Qwen2.5 model](https://github.com/QwenLM/Qwen2.5) for caption refinement
