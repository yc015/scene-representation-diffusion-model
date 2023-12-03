import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms import functional

from baukit import Trace, TraceDict

from transformers import CLIPTextModel, CLIPTokenizer, logging
from modified_diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from torch import autocast

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import ipywidgets

import pandas as pd

import os
import pickle

from probe_src.vis_partially_denoised_latents import generate_image, _init_models

import time
# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tic, toc = (time.time, time.time)

# Reproducibility
import random

from probe_src.probe_depth_datasets import ProbeOSDataset, threshold_target
from probe_src.probe_utils import dice_coeff, weighted_f1, ModuleHook, clear_dir
from probe_src.probe_models import probeLinearDense
from probe_src.depth_intervention_utils import make_counterfactual_label
from probe_src.depth_intervention_utils import load_classifiers, generate_image_with_modified_internal_rep
from collections import OrderedDict

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from intervention_config import getConfig

args = getConfig()


def main(args):
    output_path = args.output_dir
    
    # Initiating the Stable diffusion model
    logging.set_verbosity_error()

    vae_pretrained="CompVis/stable-diffusion-v1-4"
    CLIPtokenizer_pretrained="openai/clip-vit-large-patch14"
    CLIPtext_encoder_pretrained="openai/clip-vit-large-patch14"
    denoise_unet_pretrained="CompVis/stable-diffusion-v1-4"

    vae, tokenizer, text_encoder, unet, scheduler = _init_models(vae_pretrained=vae_pretrained,
                                                                 CLIPtokenizer_pretrained=CLIPtokenizer_pretrained,
                                                                 CLIPtext_encoder_pretrained=CLIPtext_encoder_pretrained,
                                                                 denoise_unet_pretrained=denoise_unet_pretrained)
    
    # Read in the prompt dataset
    train_split_prompts_seeds = pd.read_csv("train_split_prompts_seeds.csv", encoding = "ISO-8859-1")
    test_split_prompts_seeds = pd.read_csv("test_split_prompts_seeds.csv", encoding = "ISO-8859-1")
    combo_df = pd.concat([train_split_prompts_seeds, test_split_prompts_seeds])
    
    # Load in the generation seed
    dataset_path = "datasets/images/"
    files = os.listdir(dataset_path)
    files = [file for file in files if file.endswith(".png")]
    prompt_indexes = [int(file[file.find("prompt_")+7:file.find("_seed")]) for file in files]
    sample_seeds = [int(file[file.find("seed_")+5:file.find(".png")]) for file in files]

    # Uncomment the code below if we only want to test the intervention on the test samples
    prompt_indexes = test_split_prompts_seeds.prompt_inds.copy()
    sample_seeds = test_split_prompts_seeds.seeds.copy()

    # Optimization Settings
    # Use the linear classifier without bias term for intervention
    weights_type = ""
    # Load in the probing classifiers
    classifier_dicts = {}
    for step in range(15):
        classifier_dicts[f"step_{step}"] = load_classifiers(step, weights_type=weights_type)

    all_layers = list(classifier_dicts["step_0"].keys())
    chosen_layers = all_layers[7:]
    
    # Optimization rate
    lr = 5e-3
    # loss
    loss_func = nn.CrossEntropyLoss()
    # Intervened the first 5 denoising steps
    at_steps = [i for i in range(0, 5)]
    # Number of optimization epochs when modifying an internal representation
    max_epochs = [128] * 15

    # Intervention Settings
    # Translation range
    t_range_h = [(-120, -90), (90, 120)]
    t_range_v = [(-120, -90), (90, 120)]

    shift_h_lower_range = np.arange(t_range_h[0][0], t_range_h[0][1])
    shift_h_upper_range = np.arange(t_range_h[1][0], t_range_h[1][1])
    shift_h_range = np.concatenate([shift_h_lower_range, shift_h_upper_range])

    shift_v_lower_range = np.arange(t_range_v[0][0], t_range_v[0][1])
    shift_v_upper_range = np.arange(t_range_v[1][0], t_range_v[1][1])
    shift_v_range = np.concatenate([shift_v_lower_range, shift_v_upper_range])

    # Number of intervention trails
    num_trials = 5

    # Define output path
    dataset_path = "datasets/images"
    
    # Create the output directory
    figure_dir = os.path.join(output_path, "figures/")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    modified_image_dir = os.path.join(output_path, "modified_output/")
    if not os.path.exists(modified_image_dir):
        os.makedirs(modified_image_dir)
    modified_target_dir = os.path.join(output_path, "modified_target/")
    if not os.path.exists(modified_target_dir):
        os.makedirs(modified_target_dir)

    print("Intervened Layers")
    for layer_name in chosen_layers:
        print(layer_name)

    print("\nAt step:")
    print(at_steps)
    
    # Reproducibility
    np.random.seed(123)
    for ind in range(len(prompt_indexes)):
        # Read in the prompt and seeds for synthesizing original images
        prompt_ind = prompt_indexes[ind]
        prompt = combo_df.loc[combo_df['prompt_inds'] == prompt_ind]["prompts"].item()
        seed_num = sample_seeds[ind]

        # Read in the salient object mask of original image
        target = plt.imread(f"mask/images/prompt_{prompt_ind}_seed_{seed_num}.png") > 0

        # Read in the original image output
        ori_image = plt.imread(os.path.join(dataset_path, f"prompt_{prompt_ind}_seed_{seed_num}.png"))[..., :3]

        # Sample random translations
        done_translations = []
        for trial in range(num_trials):
            translation = [np.random.choice(shift_h_range),
                           np.random.choice(shift_v_range)]
            # If a sample is repeated
            while translation in done_translations:
                # Redo the sampling until get a different one
                translation = [np.random.choice(shift_h_range),
                               np.random.choice(shift_v_range)]
            done_translations.append(translation)

            rotation = 0

            # Make the modified salient object mask for intervention
            cf_target = make_counterfactual_label(target, translate=translation, angle=rotation)

            # Intervened the model with respect to the modified salient object mask
            image = generate_image_with_modified_internal_rep(prompt, seed_num,
                                                              text_encoder=text_encoder,
                                                              tokenizer=tokenizer,
                                                              unet=unet,
                                                              scheduler=scheduler,
                                                              vae=vae,
                                                              modified_layer_names=chosen_layers,
                                                              at_steps=at_steps,
                                                              lr=lr,
                                                              max_epochs=max_epochs,
                                                              classifier_dicts=classifier_dicts,
                                                              cf_target=cf_target,
                                                              loss_func=loss_func)

            saved_filename = f"prompt_{prompt_ind}_seed_{seed_num}_th_{translation[0]}_tv_{translation[1]}.png"
            
            # Save the modified label
            plt.imsave(os.path.join(modified_target_dir, saved_filename), cf_target.cpu().detach().numpy())
            
            # Save the intervened output
            plt.imsave(os.path.join(modified_image_dir, saved_filename), image)

            plt.ioff()

            fig, ax = plt.subplots(1, 6, figsize=(18, 3.2), sharey=True)
            ax[0].set_title(r"Original Output $f(x)$")
            ax[0].imshow(ori_image)

            ax[1].set_title(r"Original Label $z$")
            ax[1].imshow(target, cmap="gray")

            ax[2].imshow(ori_image)
            ax[2].imshow(target, alpha=0.5)

            ax[3].set_title(r"CF Output $f(\tilde{x})$")
            ax[3].imshow(image)

            ax[4].set_title(r"Counterfactual Label $\tilde{z}$")
            ax[4].imshow(cf_target, cmap="gray")

            ax[5].imshow(image)
            ax[5].imshow(cf_target, alpha=0.5)

            plt.suptitle(f"{prompt[:60]}", fontsize=12)

            plt.savefig(os.path.join(figure_dir, saved_filename), bbox_inches="tight", dpi=90)
            plt.close()
    
if __name__ == '__main__':
    main(args)  
    