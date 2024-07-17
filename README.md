# Project Overview

This project demonstrates how to generate images using Stable Diffusion XL with optional prompt enhancement via Gemini's API. The code includes configurable settings and the ability to save/load these settings from a JSON file.

## Prerequisites

- Python 3.8 or higher (Note: For using `torch.compile`, Python 3.11 or lower is required)
- `venv` or `conda` for creating a virtual environment

## Setup

### Using `venv`

1. **Create a virtual environment:**

    ```bash
    python -m venv civitai_env
    ```

    

2. **Activate the virtual environment:**

    - On Windows:

        ```bash
        civitai_env\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source civitai_env/bin/activate
        ```
       ```fish
        source civitai_env/bin/activate.fish
        ```

3. **Install the required libraries:**

    ```bash
    pip install torch diffusers accelerate pillow google-generativeai transformers
    ```

### Using `conda`

1. **Create a conda environment:**

    ```bash
    conda create --name civitai_env python=3.11
    ```

2. **Activate the conda environment:**

    ```bash
    conda activate civitai_env
    ```

3. **Install the required libraries:**

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install diffusers accelerate pillow google-generativeai transformers
    ```

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/al-swaiti/SDXL-LOCAL
    cd SDXL-LOCAL
    ```

2. **Run the script:**

    ```bash
    python main.py
    ```

## Configuration

The script uses a configuration file (`config.json`) "auto created after first run"to store user inputs. If the file doesn't exist, default values are used. The script will prompt you to enter the following details:

- Model location
- Prompt
- Negative prompt
- Width and height of the image
- Number of inference steps
- CFG scale
- Seed
- Gemini API key (optional)

You can also provide these inputs through the command line. If no input is provided, the default values or the values from the configuration file will be used.

### Enhancing Prompts with Gemini's API

To get professional prompt results, you can use the Gemini API. If you wish to use this feature, insert your Gemini API key when prompted or include it in the configuration file.

## Notes

- If you want to use `torch.compile`, ensure you are using Python 3.11 or lower. and enable ( pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
) line.
- The generated image will be saved with a timestamp-based filename.
- for low gpu disable ( pipe.to("cuda")) and enable (pipe.enable_sequential_cpu_offload())

  follow me on
  https://civitai.com/user/AbdallahAlswa80

  watch me on
  https://www.deviantart.com/abdallahalswaiti

  connect me on
  https://www.linkedin.com/in/abdallah-issac/

  try  my model
  https://civitai.com/models/130664


