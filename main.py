import json
import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
import accelerate
from PIL import Image
import google.generativeai as genai
import random
from datetime import datetime

CONFIG_FILE = "config.json"

# Function to get user inputs or load from file
def get_or_load_user_inputs():
    default_config = {
        "model_location": "/path/to/default/model.safetensors",
        "prompt": "default prompt",
        "negative_prompt": "nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blurry, distorted, artifacts, overexposed, underexposed, grainy, unnatural lighting, low contrast, washed out, pixelated, unnatural proportions",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg_scale": 5,
        "seed": None,  # None indicates a random seed will be used
        "gemini_api_key": ""
    }

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    else:
        config = default_config
    
    print("Press Enter to use saved values or provide new inputs:")
    model_location = input(f"Model location [{config['model_location']}]: ").strip() or config['model_location']
    prompt = input(f"Prompt [{config['prompt']}]: ").strip() or config['prompt']
    negative_prompt = input(f"Negative prompt [{config['negative_prompt']}]: ").strip() or config['negative_prompt']
    width = input(f"Width of the image [{config['width']}]: ").strip() or config['width']
    height = input(f"Height of the image [{config['height']}]: ").strip() or config['height']
    steps = input(f"Number of inference steps [{config['steps']}]: ").strip() or config['steps']
    cfg_scale = input(f"CFG scale [{config['cfg_scale']}]: ").strip() or config['cfg_scale']
    seed = input(f"Seed [{config['seed']}]: ").strip() or config['seed']
    gemini_api_key = input(f"Gemini API key (optional) [{config.get('gemini_api_key', '')}]: ").strip() or config.get('gemini_api_key')
    
    # Convert inputs to proper types, using default values if inputs are empty
    width = int(width) if width else config['width']
    height = int(height) if height else config['height']
    steps = int(steps) if steps else config['steps']
    cfg_scale = float(cfg_scale) if cfg_scale else config['cfg_scale']
    seed = int(seed) if seed else random.randint(0, 2**32 - 1)
    
    # Save updated values back to config file
    save_user_inputs(model_location, prompt, negative_prompt, width, height, steps, cfg_scale, seed, gemini_api_key=gemini_api_key)
    
    return (
        model_location,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg_scale,
        seed,
        gemini_api_key
    )

# Function to save user inputs to a file
def save_user_inputs(model_location, prompt, negative_prompt, width, height, steps, cfg_scale, seed, gemini_api_key=None):
    config = {
        "model_location": model_location,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "gemini_api_key": gemini_api_key,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

# Function to enhance prompt using Gemini's API with specific instructions
def enhance_prompt_with_gemini(prompt, api_key):
    try:
        if api_key:
            genai.configure(api_key=api_key)

            # Construct the specific instruction prompt
            instruction = f"You are Professor of creating a prompt for Stable Diffusion to generate an image. Write the phrases for art styles, art movements, artists names, light and depth effects for this {prompt}. Make this prompt the best ever, only response with prompt itself in one paragraph."

            # Generate content using Gemini API with the instruction prompt
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(instruction)
            enhanced_prompt = response.text.strip()
            return enhanced_prompt
        else:
            return prompt
    except Exception as e:
        print(f"Failed to enhance prompt with Gemini API: {e}")
        return prompt  # Fallback to original prompt on error

# Main script logic
def main():
    while True:
    

        model_location, prompt, negative_prompt, width, height, steps, cfg_scale, seed, gemini_api_key = get_or_load_user_inputs()

        # Validate model path
        if not os.path.exists(model_location):
            print(f"Error: Model file not found at {model_location}")
            return

        # Enhance the prompt
        enhanced_prompt = enhance_prompt_with_gemini(prompt, gemini_api_key)
        print("-----------------------------------------------------------------")
        print(f"Model Location: {model_location}")
        print(f"Original Prompt: {prompt}")
        print(f"Enhanced Prompt: {enhanced_prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Steps: {steps}")
        print(f"CFG Scale: {cfg_scale}")
        print(f"Seed: {seed}")
        print(f"Gemini API Key: {gemini_api_key}")
        print("-----------------------------------------------------------------")
        
        input("Press Enter to continue with the image generation...")

        vae = AutoencoderTiny.from_pretrained(
        'madebyollin/taesdxl',
        use_safetensors=True,
        torch_dtype=torch.float16,
        )

        pipe = StableDiffusionXLPipeline.from_single_file(
            model_location,
            torch_dtype=torch.float16,
            use_safetensors=True,
            vae=vae
        )

        
        
        pipe.to("cuda")
        
        #torch.compile increase speed 20%-30% , work with python <= 3.11 , and little more time for first image creation 
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


        # for low gpu disable pipe.to("cuda") and enable this line below
        # pipe.enable_sequential_cpu_offload()
        

        # Set the seed for reproducibility
        generator = torch.manual_seed(seed)

        # Run the pipeline with user inputs and enhanced prompt
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            output_type="pil",
        ).images[0]

        # Save the generated image with a timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"generated_image_{timestamp}.png"
        image.save(image_filename)

        cont = input("Do you want to generate another image? (y/n): ").strip().lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()
