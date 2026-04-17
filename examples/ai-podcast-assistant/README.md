# AI Podcast Assistant

A comprehensive toolkit for generating detailed notes, summary, and translation of podcast content using the Phi-4-Multimodal LLM with NVIDIA NIM Microservices.

## Overview

This repository contains a Jupyter notebook that demonstrates a complete workflow for processing podcast audio:

1. **Notes generation**: Convert spoken content from podcasts into detailed text notes
2. **Summarization**: Generate concise summaries of the transcribed content
3. **Translation**: Translate both the transcription and summary into different languages

The implementation leverages the powerful Phi-4-Multimodal LLM (5.6B parameters) through NVIDIA's NIM Microservices, enabling efficient processing of long-form audio content.

Learn more about the model [here](https://developer.nvidia.com/blog/latest-multimodal-addition-to-microsoft-phi-slms-trained-on-nvidia-gpus/).

## Features

- **Long Audio Processing**: Automatically chunks long audio files for processing
- **Detailed Notes Generation**: Creates well-formatted, detailed notes from audio content
- **Summarization**: Generates concise summaries capturing key points
- **Translation**: Translates content to multiple languages while preserving formatting
- **File Export**: Saves results as text files for easy sharing and reference

## Requirements

- Python 3.10–3.12 (**Python 3.13+ is not supported** — `pydub` depends on `audioop` which was removed in 3.13)
- Jupyter Notebook or JupyterLab
- NVIDIA API Key (see [Installation](#installation) section for setup instructions)
- Required Python packages:
  - requests
  - base64
  - pydub
  - Pillow (PIL)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/NVIDIA/GenerativeAIExamples.git
   cd GenerativeAIExamples/community/ai-podcast-assistant
   ```

2. Set up your NVIDIA API key:
   - Sign up for [NVIDIA NIM Microservices](https://build.nvidia.com/explore/discover?signin=true)
   - Generate an [API key](https://build.nvidia.com/microsoft/phi-4-multimodal-instruct?api_key=true)
   - Replace the placeholder in the notebook with your API key


## Example Output

The notebook generates:

1. **Detailed Notes**: Bullet-pointed notes capturing the main content of the podcast
2. **Summary**: A concise paragraph summarizing the key points
3. **Translation**: The notes and summary translated to your chosen language

All outputs are saved as text files for easy reference and sharing.

## Model Details

The Phi-4-Multimodal LLM used in this project has the following specifications:
- **Parameters**: 5.6B
- **Inputs**: Text, Image, Audio
- **Context Length**: 128K tokens
- **Training Data**: 5T text tokens, 2.3M speech hours, 1.1T image-text tokens
- **Supported Languages**: Multilingual text and audio (English, Chinese, German, French, etc.)


## Virtual Environment Setup (Recommended)

Running the notebook inside a dedicated virtual environment avoids dependency conflicts and keeps the kernel isolated from your system Python.

### Steps

1. **Create the virtual environment** inside the project folder using Python 3.11 or 3.12:
   ```bash
   cd GenerativeAIExamples/community/ai-podcast-assistant
   python3.11 -m venv envpodc
   ```
   > **Important:** Do not use Python 3.13 or 3.14. The `pydub` library depends on `audioop`, which was removed in Python 3.13. If you see `ModuleNotFoundError: No module named 'audioop'` or `pyaudioop`, recreate the venv with Python 3.11 or 3.12.

2. **Activate it:**
   ```bash
   # macOS / Linux
   source envpodc/bin/activate

   # Windows
   envpodc\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install pydub requests Pillow jupyterlab
   ```

4. **Verify JupyterLab is available in the venv:**
   ```bash
   which jupyter
   ```
   The output must point to `envpodc/bin/jupyter`. If it points to a system path, the venv is not active — re-run step 3.


5. **Launch JupyterLab:**
   ```bash
   jupyter lab
   ```
   Then open `ai-podcast-assistant-phi4-mulitmodal.ipynb` from the file browser on the left.

6. **Optional – install ffmpeg** (required by `pydub` to read/write audio files):
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu / Debian
   sudo apt install ffmpeg

   # Windows (via Chocolatey)
   choco install ffmpeg
   ```

> **Note:** If you close the terminal, re-activate the venv (`source envpodc/bin/activate`) before launching Jupyter again.

### [More examples ](https://github.com/NVIDIA/GenerativeAIExamples/tree/main)
