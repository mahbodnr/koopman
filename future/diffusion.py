import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

def run_diffusion_pipeline(
    prompt: str,
    num_inference_steps: int = 20,
    save_intermediates: bool = True,
    output_dir: str = "intermediate_images",
    seed: int = 42
):
    """
    Run a small diffusion model for the given prompt and save intermediate steps.
    
    Args:
        prompt (str): The input text prompt
        num_inference_steps (int): Number of denoising steps
        save_intermediates (bool): Whether to save intermediate images
        output_dir (str): Directory to save intermediate images
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Final image and list of intermediate images if save_intermediates=True
    """
    # Set random seed for reproducibility
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    # Create output directory if needed
    if save_intermediates:
        import os
        os.makedirs(output_dir, exist_ok=True)
    # Load small pretrained model
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd",
        torch_dtype=dtype,
    )
    # Use DDIM scheduler for better control over steps
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe = pipe.to(device)

    # Custom denoising step function
    def custom_denoise_step(i, t, latents):
        # Get model output
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_embeddings)["sample"]
        
        # Scheduler step
        latents = pipe.scheduler.step(
            noise_pred,
            t,
            latents,
        ).prev_sample
        
        if save_intermediates:
            # Decode latents to image
            with torch.no_grad():
                image = pipe.decode_latents(latents)
                image = pipe.numpy_to_pil(image)[0]
                image.save(f"{output_dir}/step_{i:03d}.png")
                image.close()
        
        return latents
    
    # Encode text
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    
    # Initialize random latents
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        device=device,
        dtype=dtype,
    )
    
    latents_history = [latents.clone().detach()]
    # Denoise
    for i, t in enumerate(tqdm(pipe.scheduler.timesteps)):
        latents = custom_denoise_step(i, t, latents)
        latents_history.append(latents.clone().detach())
    
    # Decode final image
    with torch.no_grad():
        final_image = pipe.decode_latents(latents)
        final_image = pipe.numpy_to_pil(final_image)[0]
    
    return final_image, torch.stack(latents_history)

# Example usage
if __name__ == "__main__":
    prompt = "a beautiful sunset over mountains"
    with torch.no_grad():
        final_image, intermediates = run_diffusion_pipeline(
            prompt=prompt,
            num_inference_steps=10,
            save_intermediates=True
        )
    
    # Save final image
    final_image.save("final_image.png")
    print(intermediates.shape)