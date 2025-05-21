import os
import json
import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from diffusion import Diffusion
from dataloader import get_tokenizer
from classifier import Classifier
from torchvision.utils import save_image 

@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(config: DictConfig):
    """
    Generate CIFAR samples for a given class using a trained diffusion model.
    """
    # Reproducibility
    print(config)
    torch.manual_seed(config.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Load tokenizer and model
    tokenizer = get_tokenizer(config)
    model = Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        logger=False
    )
    model.eval()

    # Generate samples
    samples = []
    png_dir = config.eval.generated_samples_path
    os.makedirs(png_dir, exist_ok=True)
    sample_idx = 0

    for _ in tqdm(range(config.sampling.num_sample_batches), desc="Generating batches"):
        import random
        model.config.guidance.condition = random.randint(0, 9)
        sample = model.sample()  # Assume shape: (batch, C, H, W) or (C, H, W)
        decoded_samples = tokenizer.batch_decode(sample)
        samples.extend(decoded_samples)
        decoded_samples = decoded_samples.to(torch.float32) / 255.0  # Normalize to [0, 1]

        # Save each sample in the batch as PNG
        if isinstance(decoded_samples, torch.Tensor):
            if decoded_samples.dim() == 3:  # Single image
                save_image(decoded_samples, os.path.join(png_dir, f"sample_{sample_idx:05d}.png"))
                sample_idx += 1
            elif decoded_samples.dim() == 4:  # Batch of images
                for img in decoded_samples:
                    save_image(img, os.path.join(png_dir, f"sample_{sample_idx:05d}.png"))
                    sample_idx += 1
        else:
            print("Sample is not a torch.Tensor. Please adapt saving logic.")
    print(f"PNG images saved to: {png_dir}")

if __name__ == "__main__":
    main()