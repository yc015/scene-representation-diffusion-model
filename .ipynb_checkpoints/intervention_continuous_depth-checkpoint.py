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

from probe_src.probe_depth_datasets import ProbeOSDataset, min_max_norm_target, norm_target, norm_intervention_target
from probe_src.probe_utils import ModuleHook, clear_dir
from probe_src.probe_models import probeLinearDense
from probe_src.continuous_depth_intervention_utils import load_classifiers_continuous_depth
from probe_src.continuous_depth_intervention_utils import generate_image_with_modified_internal_rep
from probe_src.continuous_depth_intervention_utils import make_counterfactual_label
from collections import OrderedDict

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from intervention_config import getConfig

args = getConfig()

def main(args):
    output_path = args.output_dir
    
    # Initiating the Stable Diffusion model
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
    # Use linear regressor without bias term for intervention
    weights_type="_linear_no_bias_unsmoothed"
    # Intervened the first three steps
    at_steps = [i for i in range(0, 3)]
    # Load in the probing regressors
    classifier_dicts = {}
    for step in at_steps:
        classifier_dicts[f"step_{step}"] = load_classifiers_continuous_depth(step,
                                                                             weights_type=weights_type,)

    all_layers = list(classifier_dicts[f"step_0"].keys())
    chosen_layers = all_layers[0:]
    
    # Optimization rate
    lr = 5e-3
    # Loss
    loss_func = nn.HuberLoss()
    smoothness_loss_func = None
    # Number of optimization epoch at each intervened step
    max_epochs = [128] * 3
    
    # Intervention Hyperparameters
    # Depth label translation range
    t_range_h = [(-120, -90), (90, 120)]
    t_range_v = [(-120, -90), (90, 120)]
    
    shift_h_lower_range = np.arange(t_range_h[0][0], t_range_h[0][1])
    shift_h_upper_range = np.arange(t_range_h[1][0], t_range_h[1][1])
    shift_h_range = np.concatenate([shift_h_lower_range, shift_h_upper_range])

    shift_v_lower_range = np.arange(t_range_v[0][0], t_range_v[0][1])
    shift_v_upper_range = np.arange(t_range_v[1][0], t_range_v[1][1])
    shift_v_range = np.concatenate([shift_v_lower_range, shift_v_upper_range])

    num_trials = 5

    # Define output path
    dataset_path = "datasets/images"
    
    # Create the output dir
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
    np.random.seed(1)
    for ind in range(len(prompt_indexes)):
        # Read in the prompt and seeds used in generating original image
        prompt_ind = prompt_indexes[ind]
        prompt = combo_df.loc[combo_df['prompt_inds'] == prompt_ind]["prompts"].item()
        seed_num = sample_seeds[ind]

        # read in the original depth label
        with open(f"datasets/depth_gt/prompt_{prompt_ind}_seed_{seed_num}.pkl", "rb") as infile:  
            target = norm_target(pickle.load(infile))

        # Read in the original synthesized image
        ori_image = plt.imread(os.path.join(dataset_path, f"prompt_{prompt_ind}_seed_{seed_num}.png"))[..., :3]

        # Accumulate translation trial
        done_translations = []
        for trial in range(num_trials):
            # Get random horizontal and vertical translation
            rotation = 0
            translation = [np.random.choice(shift_h_range),
                           np.random.choice(shift_v_range)]
            # If translation is repeated
            while translation in done_translations:
                # Redo the sampling
                translation = [np.random.choice(shift_h_range),
                               np.random.choice(shift_v_range)]
            # Until accumulates num_trails of different translation
            done_translations.append(translation)
            
            # Make the modified label for intervention
            cf_target = make_counterfactual_label(target, translate=[translation[0], 0], angle=rotation)
            
            # Fill in the empty area outside translated depth map with edge values
            if translation[0] < 0:
                cf_target[:, translation[0]:] = cf_target[:, translation[0] - 1].unsqueeze(1)
            else:
                cf_target[:, :translation[0]] = cf_target[:, translation[0] + 1].unsqueeze(1)

            if translation[1] != 0:
                cf_target = make_counterfactual_label(cf_target.cpu().detach().numpy(), 
                                                      translate=[0, translation[1]])
                if translation[1] < 0:
                    cf_target[translation[1]:, :] = cf_target[translation[1] - 1, :].unsqueeze(0)
                else:
                    cf_target[:translation[1], :] = cf_target[translation[1] + 1, :].unsqueeze(0)
            
            cf_target = norm_intervention_target(cf_target)
            
            # Intervened the model's output with respect to the modified depth label
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
                                                              loss_func=loss_func,
                                                              smoothness_loss_func=smoothness_loss_func,
                                                              image=ori_image)
            
            
            saved_filename = f"prompt_{prompt_ind}_seed_{seed_num}_th_{translation[0]}_tv_{translation[1]}"
            
            # Save the modified target
            with open(os.path.join(modified_target_dir, 
                                   f"{saved_filename}.pkl"), "wb") as outfile:
                      pickle.dump(cf_target.cpu().detach().numpy(), outfile)
            
            # Save the intervened output
            plt.imsave(os.path.join(modified_image_dir, f"{saved_filename}.png"), image)

            plt.ioff()

            fig, ax = plt.subplots(1, 6, figsize=(18, 3.2), sharey=True)
            ax[0].set_title(r"Original Output $f(x)$")
            ax[0].imshow(ori_image)

            ax[1].set_title(r"Original Label $z$")
            ax[1].imshow(target, cmap="turbo")

            ax[2].imshow(ori_image)
            ax[2].imshow(target, alpha=0.5)

            ax[3].set_title(r"CF Output $f(\tilde{x})$")
            ax[3].imshow(image)

            ax[4].set_title(r"Counterfactual Label $\tilde{z}$")
            ax[4].imshow(cf_target, cmap="turbo")

            ax[5].imshow(image)
            ax[5].imshow(cf_target, alpha=0.5, cmap="turbo")

            plt.suptitle(f"{prompt[:60]}", fontsize=12)

            plt.savefig(os.path.join(figure_dir, f"{saved_filename}.jpg"), bbox_inches="tight", dpi=90)
            plt.close()
    
if __name__ == '__main__':
    main(args)  