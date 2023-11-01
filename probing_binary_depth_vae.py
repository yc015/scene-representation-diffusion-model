import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from transformers import CLIPTextModel, CLIPTokenizer
from modified_diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from torch import autocast

from PIL import Image

import numpy as np

from typing import Tuple, List, Any, Union
import matplotlib.pyplot as plt

import os
import time
import pandas as pd
import pickle
from collections import OrderedDict

from probe_src.vis_partially_denoised_latents import generate_image, _init_models

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tic, toc = (time.time, time.time)

import copy

from probe_src.probe_depth_datasets import ProbeOSDataset, threshold_target
from probe_src.probe_utils import dice_coeff, weighted_f1, plt_test_results, train, test, ModuleHook, clear_dir
from probe_src.probe_models import probeLinearDense

# Reproducibility
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


input_dims_dict = {"mid_0": 512,
                   "up_0": 512,
                   "up_1": 512,
                   "up_2": 256,
                   "up_3": 128}

scale_dict = {"mid_0": 8,
              "up_0": 8,
              "up_1": 4,
              "up_2": 2,
              "up_3": 1}


from probing_depth_config import getConfig

args = getConfig()

def main(args):
    # Get probing arguments
    at_step = args.step
    output_dir_name = args.output_dir
    layer_name = args.layer_name
    block_type = args.block_type
    postfix = args.postfix
    
    print("At denoising step", at_step + 1)
    
    # Set layer type for different blocks (convolutional layer for ResNets, linear layer for Transformers)
    if block_type == "resnets":
        torch_layer_type = torch.nn.Conv2d
    elif block_type == "attentions":
        torch_layer_type = torch.nn.Linear
    
    # Create the directory for saving checkpoints and accuracy
    probe_checkpoints_dir = f"probe_checkpoints/large_syn_dataset_vae_step/at_step_{at_step}/"
    if not os.path.exists(probe_checkpoints_dir):
        os.makedirs(probe_checkpoints_dir)

    probe_accuracy_dir = f"probe_accuracy/large_syn_dataset_vae_step/at_step_{at_step}/"
    if not os.path.exists(probe_accuracy_dir):
        os.makedirs(probe_accuracy_dir)
    
    # Read in the prompts and seeds for generating training and test sets images
    train_split_prompts_seeds = pd.read_csv("train_split_prompts_seeds.csv")
    test_split_prompts_seeds = pd.read_csv("test_split_prompts_seeds.csv")
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
    
    # Save partially denoised latents for probing VAE self attention
    save_latents(text_encoder=text_encoder, tokenizer=tokenizer, 
                 unet=unet, scheduler=scheduler,
                 vae=vae, step=at_step + 1, 
                 prompt_indexes=prompt_indexes, sample_seeds=sample_seeds,
                 combo_df=combo_df)
    
    # Get train set prompts id
    train_prompt_indexes = train_split_prompts_seeds.prompt_inds.tolist()
    # Get train set generation seeds
    train_sample_seeds = train_split_prompts_seeds.seeds.tolist()

    # Get test set prompts id
    test_prompt_indexes = test_split_prompts_seeds.prompt_inds.tolist()
    # Get test set generation seeds
    test_sample_seeds = test_split_prompts_seeds.seeds.tolist()
    
    # Probe the representation in VAE's Decoder (bottleneck + upsampling blocks)
    for block in ["mid", "up"]:
        if block == "down":
            i_s = 0
            i_e = 3
            layer_range = 3
            if block_type == "attentions":
                continue
        elif block == "up":
            i_s = 0
            i_e = 4
            # i_e = 3
            layer_range = 3
            if block_type == "attentions":
                continue
        elif block == "mid":
            i_s = 0
            i_e = 1
            if block_type == "resnets":
                layer_range = 2
            else:
                layer_range = 1

        for block_ind in range(i_s, i_e):
            data_path = "datasets"
            for layer_ind in range(layer_range):
                # Save the intermediate output of VAE layers
                # They are used as the input to the probing classifiers
                save_internal_output(block=block, block_ind=block_ind, block_type=block_type, 
                                     layer_ind=layer_ind, layer_name=layer_name, output_dir_name=output_dir_name,
                                     subset_index=train_prompt_indexes,
                                     vae=vae, torch_layer_type=torch_layer_type, combo_df=combo_df,
                                     prompt_indexes=prompt_indexes, sample_seeds=sample_seeds) 

                dataset_path = "internal_repres_vae/"
                dataset_path += f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"
                layer = f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"
                dataset = ProbeOSDataset("datasets/images/", 
                                         os.path.join(data_path, dataset_path),
                                         "mask/images/",
                                         target_transform=threshold_target,
                                         vae_inter_rep=True)

                # Create the probing classifier
                input_dim = input_dims_dict[f"{block}_{block_ind}"]
                scale = scale_dict[f"{block}_{block_ind}"]
                probe = probeLinearDense(input_dim, 2, scale, use_bias=False).to(torch_device)

                # Reproducibility
                generator = torch.manual_seed(100)

                # Create the dataset
                with open("train_indices.pkl", "rb") as infile:
                    train_indices = pickle.load(infile)

                with open("test_indices.pkl", "rb") as infile:
                    test_indices = pickle.load(infile)

                # Use the pre-split train test sets
                training_data = torch.utils.data.Subset(dataset, train_indices)
                test_data = torch.utils.data.Subset(dataset, test_indices)
                
                train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
                test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

                # Set up optimizer
                optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

                # Train the probe
                max_epoch = 30
                loss_func = nn.CrossEntropyLoss()
                min_loss = 1e6

                for epoch in range(1, max_epoch + 1):
                    verbosity = False
                    if epoch == max_epoch:
                        verbosity = True
                        print(f"\n{block} Block {block_ind} Layer {layer_ind} {layer_name}")
                    # Get the train results from training of each epoch
                    train_results = train(probe, torch_device, train_dataloader, optimizer, 
                                          epoch, loss_func=loss_func, verbose_interval=None,
                                          metrics=dice_coeff, head=None, verbose=verbosity)

                dataset_path = os.path.join(data_path, dataset_path)
                clear_dir(dataset_path, file_extention=".pkl")

                save_internal_output(block=block, block_ind=block_ind, block_type=block_type,
                                     layer_ind=layer_ind, layer_name=layer_name, output_dir_name=output_dir_name,
                                     subset_index=test_prompt_indexes,
                                     vae=vae, torch_layer_type=torch_layer_type, combo_df=combo_df,
                                     prompt_indexes=prompt_indexes, sample_seeds=sample_seeds) 

                test_results = test(probe, torch_device, test_dataloader, loss_func=loss_func, 
                                    return_raw_outputs=True, metrics=dice_coeff, head=None, verbose=verbosity)

                # Save the test results at last epoch for evaluation
                with open(f"probe_accuracy/large_syn_dataset_vae_step/at_step_{at_step}/saved_test_results_{block}_{block_ind}_layer_{layer_ind}_{postfix}.pkl", "wb") as outfile:
                    pickle.dump(test_results[1], outfile)

                # Save the probe model's weights
                torch.save(probe.state_dict(), f"probe_checkpoints/large_syn_dataset_vae_step/at_step_{at_step}/segmentation_probe_{layer}_final.pth")

                # Plot the probing results of first 20 samples in the test set
                plt_test_results(probe, 
                                 test_dataloader, 
                                 test_data, 
                                 loss_func, head=None,
                                 save_plt=True,
                                 save_filename=f"probe_accuracy/large_syn_dataset_vae_step/at_step_{at_step}/saved_test_results_{block}_{block_ind}_layer_{layer_ind}_{postfix}.png")

                # Clear the cached internal representations
                clear_dir(dataset_path, file_extention=".pkl")   
                

def save_internal_output(block, block_ind, block_type, layer_ind, layer_name, output_dir_name,
                         vae=None, torch_layer_type=None, combo_df=None,
                         subset_index=None, prompt_indexes=None, sample_seeds=None):
    """
    Helper function for saving the intermediate output of VAE at a selected layer
    """
    data_path = "datasets"
    dataset_path = "internal_repres_vae/"
    dataset_path += f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"
    dir_path = os.path.join(data_path, dataset_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)    
    for prompt_ind, seed_num in zip(prompt_indexes, sample_seeds):
        prompt = combo_df.loc[combo_df['prompt_inds'] == prompt_ind]["prompts"].item()

        if subset_index and not (prompt_ind in subset_index):
            pass
        else:
            features = OrderedDict()
            for name, module in vae.decoder.named_modules():
                if isinstance(module, torch_layer_type):
                    features[name] = ModuleHook(module)
            
            with torch.no_grad():
                with open(os.path.join(data_path, "denoised_latents/",
                                       f"denoised_latents_{prompt_ind}_{seed_num}.pkl"), "rb") as infile:
                    latents = pickle.load(infile).to(torch_device).unsqueeze(0)

                image = vae.decode(latents)

            for feature in features.values():
                feature.close()

            if block == "mid":
                chosen_layer_name = f"mid_block.{block_type}.{layer_ind}.{layer_name}"
            else:
                chosen_layer_name = f"{block}_blocks.{block_ind}.{block_type}.{layer_ind}.{layer_name}"

            sel_output = features[chosen_layer_name].features[0]
            sel_output = sel_output.unsqueeze(0).cpu().detach()
        
        with open(os.path.join(data_path, dataset_path, f"{block}_{block_ind}_layer_{layer_ind}_{prompt_ind}_{seed_num}.pkl"), "wb") as outfile:
            if subset_index and not (prompt_ind in subset_index):
                pass
            else:
                pickle.dump(sel_output, outfile)
                

def save_latents(tokenizer, text_encoder, unet, scheduler, vae, step=None, 
                 prompt_indexes=None, sample_seeds=None, combo_df=None):
    """
    Helper function for saving the partially denoised latent vectors output at the denosing steps `step`
    """
    data_path = "datasets"
    dataset = f"denoised_latents"
    dir_path = os.path.join(data_path, dataset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  
        
    clear_dir(os.path.join(data_path, dataset), file_extention=".pkl")

    for prompt_ind, seed_num in zip(prompt_indexes, sample_seeds):
        prompt = combo_df.loc[combo_df['prompt_inds'] == prompt_ind]["prompts"].item()

        latents = generate_image(prompt, seed_num, num_inference_steps=15,
                                 net=unet, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, vae=vae,
                                 return_latents=True, stop_at_step=step)

        sel_output = latents[0].cpu().detach()
        with open(os.path.join(data_path, dataset, f"denoised_latents_{prompt_ind}_{seed_num}.pkl"), "wb") as outfile:
            pickle.dump(sel_output, outfile)
                        
if __name__ == '__main__':
    main(args)   
