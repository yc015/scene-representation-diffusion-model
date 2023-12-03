import torch
from torch import nn

from transformers import CLIPTextModel, CLIPTokenizer
from modified_diffusers import AutoencoderKL, LMSDiscreteScheduler
from modified_diffusers import UNet2DConditionModel
from torch import autocast

import numpy as np

import matplotlib.pyplot as plt

import os
import time
import pickle
from collections import OrderedDict

from probe_src.probe_utils import ModuleHook

from probe_src.vis_partially_denoised_latents import generate_image, _init_models

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tic, toc = (time.time, time.time)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from probe_src.probe_depth_datasets import ProbeDEDataset, min_max_norm_target, min_max_norm_image, scale_and_norm
from probe_src.probe_utils import train_continuous_depth, test_continuous_depth, plt_test_results_continuous_depth, clear_dir
from probe_src.depth_loss import ScaleAndShiftInvariantLoss, InverseDepthSmoothnessLoss
from probe_src.probe_models import probeLinearDense
from scipy.stats import spearmanr, pearsonr

import copy

# Reproducibility
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import pandas as pd


input_dims_dict = {"down_0": 320,
                   "down_1": 640,
                   "down_2": 1280,
                   "up_1": 1280,
                   "up_2": 640,
                   "up_3": 320,
                   "mid_0": 1280}

scale_dict = {"down_0": 8,
              "down_1": 16,
              "down_2": 32,
              "up_1": 32,
              "up_2": 16,
              "up_3": 8,
              "mid_0": 64}


from probing_depth_config import getConfig

args = getConfig()

def main(args):
    # Get probing arguments
    at_step = args.step
    output_dir_name = args.output_dir
    layer_name = args.layer_name
    block_type = args.block_type
    postfix = args.postfix
    probe_type = args.probe_type
    normalized = args.normalized
    smoothed = args.smoothed
    lasso = True if args.lasso == "yes" else False
    l1_lambda = float(args.l1_lambda)
    
    # Set layer type for different blocks (convolutional layer for ResNets, linear layer for Transformers)
    if block_type == "resnets":
        torch_layer_type = torch.nn.Conv2d
    elif block_type == "attentions":
        torch_layer_type = torch.nn.Linear
    
    print("At denoising step", at_step + 1)
    # Create the directory for saving checkpoints and accuracy
    probe_checkpoints_dir = f"probe_checkpoints/large_syn_dataset_continuous_fully_permuted/at_step_{at_step}/"
    if not os.path.exists(probe_checkpoints_dir):
        os.makedirs(probe_checkpoints_dir)

    probe_accuracy_dir = f"probe_accuracy/large_syn_dataset_continuous_fully_permuted/at_step_{at_step}/"
    if not os.path.exists(probe_accuracy_dir):
        os.makedirs(probe_accuracy_dir)
    
    # Read in the prompt ids and seeds for generating training and test sets images
    train_split_prompts_seeds = pd.read_csv("train_split_prompts_seeds.csv", encoding = "ISO-8859-1")
    test_split_prompts_seeds = pd.read_csv("test_split_prompts_seeds.csv", encoding = "ISO-8859-1")
    combo_df = pd.concat([train_split_prompts_seeds, test_split_prompts_seeds])
    
    # Get all prompts and seeds used in generating images in the probing dataset
    dataset_path = "datasets/images/"
    files = os.listdir(dataset_path)
    files = [file for file in files if file.endswith(".png")]
    prompt_indexes = [int(file[file.find("prompt_")+7:file.find("_seed")]) for file in files]
    sample_seeds = [int(file[file.find("seed_")+5:file.find(".png")]) for file in files]
    
    # Initialize the Stable diffusion model
    vae_pretrained="CompVis/stable-diffusion-v1-4"
    CLIPtokenizer_pretrained="openai/clip-vit-large-patch14"
    CLIPtext_encoder_pretrained="openai/clip-vit-large-patch14"
    denoise_unet_pretrained="CompVis/stable-diffusion-v1-4"

    vae, tokenizer, text_encoder, unet, scheduler = _init_models(vae_pretrained=vae_pretrained,
                                                                 CLIPtokenizer_pretrained=CLIPtokenizer_pretrained,
                                                                 CLIPtext_encoder_pretrained=CLIPtext_encoder_pretrained,
                                                                 denoise_unet_pretrained=denoise_unet_pretrained)

    # Reproducibility
    torch.manual_seed(10000)
    # Randomized LDM
    permuted_unet = copy.deepcopy(unet)

    # Randomize the pretrained LDM to simulate an untrained model for baseline comparison
    with torch.no_grad():
        for name, module in permuted_unet.named_modules():
            if hasattr(module, "weight"):
                t = module.weight.clone()
                idx = torch.randperm(t.nelement())
                t = t.view(-1)[idx].view(t.size())
                module.weight = nn.Parameter(t)
    
    # Probe the representation in LDM
    for block in ["down", "mid", "up"]:
        if block == "down":
            i_s = 0
            i_e = 3
            layer_range = 2
        elif block == "up":
            i_s = 1
            i_e = 4
            layer_range = 3
        elif block == "mid":
            i_s = 0
            i_e = 1
            if block_type == "resnets":
                layer_range = 2
            else:
                layer_range = 1

        for block_ind in range(i_s, i_e):
            data_path = "datasets"

            for prompt_ind, seed_num in zip(prompt_indexes, sample_seeds):
                # Save the intermediate output of LDM self-attention layers
                # They are used as the input to the probing classifiers
                features = OrderedDict()

                # recursive hooking function
                for name, module in permuted_unet.named_modules():
                    if isinstance(module, torch_layer_type):
                        features[name] = ModuleHook(module)

                prompt = combo_df.loc[combo_df['prompt_inds'] == prompt_ind]["prompts"].item()

                # Regenerate the images in probing dataset
                image = generate_image(prompt, seed_num, num_inference_steps=15,
                                       net=permuted_unet, tokenizer=tokenizer, 
                                       text_encoder=text_encoder, scheduler=scheduler, vae=vae,
                                       stop_at_step=at_step+1)

                for feature in features.values():
                    feature.close()

                for layer_ind in range(layer_range):
                    dataset_path = "internal_repres/"
                    dataset_path += f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"

                    if block == "mid":
                        chosen_layer_name = f"mid_block.{block_type}.{layer_ind}.{layer_name}"
                    else:
                        chosen_layer_name = f"{block}_blocks.{block_ind}.{block_type}.{layer_ind}.{layer_name}"

                    sel_output = features[chosen_layer_name].features[at_step]
                    sel_output = sel_output.unsqueeze(0).cpu().detach()
                    if not os.path.exists(os.path.join(data_path, dataset_path)):
                        os.makedirs(os.path.join(data_path, dataset_path))
                    # Save the intermediate output of self-attention layer
                    with open(os.path.join(data_path, dataset_path, f"{block}_{block_ind}_layer_{layer_ind}_{prompt_ind}_{seed_num}.pkl"), "wb") as outfile:
                        pickle.dump(sel_output, outfile)

            # After collecting all internal representations (intermediate outputs)
            for layer_ind in range(layer_range):
                # Train the probe on them
                dataset_path = "internal_repres/"
                dataset_path += f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"

                # Create the probing dataset
                layer = f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"
                dataset = ProbeDEDataset("datasets/images/", 
                                         f"datasets/internal_repres/{layer}/",
                                         "datasets/depth_gt/",
                                         pre_load=True,
                                         target_transform=scale_and_norm,
                                         transform=min_max_norm_image,
                                         scale_factor=1)

                # Create the probing regressor
                input_dim = input_dims_dict[f"{block}_{block_ind}"]
                scale = scale_dict[f"{block}_{block_ind}"]
                
                weights_postfix = ""
                
                if probe_type.lower() == "linear":
                    probe = probeLinearDense(input_dim, 1, scale, use_bias=True).to(torch_device)
                    weights_postfix = ""
                elif probe_type.lower() == "linear-no-bias":
                    probe = probeLinearDense(input_dim, 1, scale, use_bias=False).to(torch_device)
                    weights_postfix = "_linear_no_bias"
                elif probe_type.lower() == "nonlinear":
                    probe = probeTwoNonLinearDense(input_dim, 1, scale, use_bias=True, 
                                                   mid_channels=input_dim // 2).to(torch_device)
                    weights_postfix = "_nonlinear"
                elif probe_type.lower() == "nonlinear-no-bias":
                    probe = probeTwoNonLinearDense(input_dim, 1, scale, use_bias=False, 
                                                   mid_channels=input_dim // 2).to(torch_device)
                    weights_postfix = "_nonlinear_no_bias"
                    
                generator = torch.manual_seed(100)

                # Use the pre-split train test sets
                with open("train_indices.pkl", "rb") as infile:
                    train_indices = pickle.load(infile)

                with open("test_indices.pkl", "rb") as infile:
                    test_indices = pickle.load(infile)

                training_data = torch.utils.data.Subset(dataset, train_indices)
                test_data = torch.utils.data.Subset(dataset, test_indices)
                train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
                test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

                # Set up the optimizer
                optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

                # Train the probe
                max_epoch = 30
                loss_func = nn.HuberLoss()
                if smoothed.lower() == "yes":
                    smooth_loss_func = InverseDepthSmoothnessLoss()
                    weights_postfix += ""
                elif smoothed.lower() == "no":
                    smooth_loss_func = None
                    weights_postfix += "_unsmoothed"
                min_loss = 1e6

                for epoch in range(1, max_epoch + 1):
                    verbosity = False
                    if epoch == max_epoch:
                        verbosity = True
                        print(f"\n{block} Block {block_ind} Layer {layer_ind} {layer_name}")
                    # Get the train results from training of each epoch
                    train_results = train_continuous_depth(probe, torch_device, train_dataloader, optimizer, 
                                                           epoch, loss_func=loss_func, verbose_interval=None,
                                                           head=None, verbosity=False, 
                                                           smooth_loss=smooth_loss_func, alpha=1)
                    test_results = test_continuous_depth(probe, torch_device, test_dataloader, loss_func=loss_func,
                                                         return_raw_outputs=verbosity, head=None, 
                                                         scheduler=None, verbosity=verbosity,
                                                         smooth_loss=smooth_loss_func, alpha=1)
                    if test_results[0] < min_loss:
                        min_loss = test_results[0]
                        torch.save(probe.state_dict(), f"probe_checkpoints/large_syn_dataset_continuous_fully_permuted/at_step_{at_step}/regression_probe_{layer}{weights_postfix}.pth")

                # Save the test results at the last epoch for evaluation
                with open(f"probe_accuracy/large_syn_dataset_continuous_fully_permuted/at_step_{at_step}/saved_test_results_{block}_{block_ind}_layer_{layer_ind}_{postfix}{weights_postfix}.pkl", "wb") as outfile:
                    pickle.dump(test_results[2], outfile)

                # Save the probe model's weights
                torch.save(probe.state_dict(), f"probe_checkpoints/large_syn_dataset_continuous_fully_permuted/at_step_{at_step}/regression_probe_{layer}_final{weights_postfix}.pth")

                # Plot the probing results of first 20 samples in the test set
                plt_test_results_continuous_depth(probe,
                                                  test_dataloader, test_data, 
                                                  loss_func, smooth_loss=smooth_loss_func,
                                                  head=None, save_plt=True,
                                                  save_filename=f"probe_accuracy/large_syn_dataset_continuous_fully_permuted/at_step_{at_step}/saved_test_results_{block}_{block_ind}_layer_{layer_ind}_{postfix}{weights_postfix}.png")

                dataset_path = os.path.join(data_path, dataset_path)
                clear_dir(dataset_path, file_extention=".pkl")


if __name__ == '__main__':
    main(args)   
