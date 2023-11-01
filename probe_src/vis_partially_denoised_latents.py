import torch
from transformers import CLIPTextModel, CLIPTokenizer
from modified_diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt

from torchvision import transforms as tfms

import numpy as np

import matplotlib.pyplot as plt

import pickle

import os

from matplotlib import animation, rc
from IPython.display import HTML


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Required Models Autoencoder decoder, tokenizer, text embedding encoder, denoising unet, scheduler
vae = None
tokenizer = None
text_encoder = None
unet = None
scheduler = None


def _init_models(vae_pretrained="CompVis/stable-diffusion-v1-4", 
                 CLIPtokenizer_pretrained="openai/clip-vit-large-patch14",
                 CLIPtext_encoder_pretrained="openai/clip-vit-large-patch14",
                 denoise_unet_pretrained="CompVis/stable-diffusion-v1-4",
                 return_attn=False,
                 return_qkv=False):
    """
    Use pretrained weights from HuggingFace
    Please make sure you download huggingface CLI and login with your credential
    
    You may also need to register for the access to pretrained stable diffusion model weights on Huggingface
    """
    
    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(vae_pretrained, subfolder="vae", use_auth_token=True)

    # Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(CLIPtokenizer_pretrained)
    text_encoder = CLIPTextModel.from_pretrained(CLIPtext_encoder_pretrained)

    # The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(denoise_unet_pretrained, subfolder="unet", use_auth_token=True, 
                                                return_attn=return_attn, return_qkv=return_qkv)

    # The noise scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # To the GPU we go!
    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)
    
    vae.eval()
    text_encoder.eval()
    unet.eval();
    
    return vae, tokenizer, text_encoder, unet, scheduler


def generate_a_dict_of_intermediate_decodings(prompt, seed_num, 
                                              height=512, width=512, 
                                              num_inference_steps=30,
                                              guidance_scale=7.5,
                                              batch_size=1,
                                              return_dict=True,
                                              save_latents=False,
                                              save_imgs=False,
                                              save_gif=False,
                                              save_final_image=False,
                                              path_latents=None,
                                              path_imgs=None,
                                              path_gif=None,
                                              path_img=None,
                                              save_dir="data"):
    """
    Generate a dictionary with 8 keys:
    prompt: A text prompt that was used to condition the image generation
    type: str
    height: Height in pixels of the generated images
    type: int
    width: Width in pixels of the generated images
    type: int
    num_inference_steps: Number of steps in denoising
    type: int
    guidance_scale: Guidance scale
    type: float
    seed_num: Seed for random generator
    type: int
    latents: A torch tensor of partially denoised latents at each step shape=[num_inference_steps, 4, 64, 64]
    type: torch.Tensor
    images: Image decoded from latent vector at each denosing step shape=[num_inference_steps, height, width, 3]
    type: numpy.ndarray
    
    return_dict: whether to return a dict of above key-value pairs
    """
    global vae, tokenizer, text_encoder, unet, scheduler
    if save_latents and (not path_latents):
        path_latents = "{:s}/latents/{:s}_{:d}.pkl".format(save_dir, prompt, seed_num).replace(" ", "_")
    if save_imgs and (not path_imgs):
        path_imgs = "{:s}/imgs/{:s}_{:d}".format(save_dir, prompt, seed_num).replace(" ", "_")
        if not os.path.exists(path_imgs):
            os.mkdir(path_imgs)
        path_imgs = path_imgs + "/{:s}_{:d}.png".format(prompt, seed_num).replace(" ", "_")
    if save_gif and (not path_gif):
        path_gif = "{:s}/gifs/{:s}_{:d}.gif".format(save_dir, prompt, seed_num).replace(" ", "_")
    if save_final_image and (not path_img):
        path_img = "{:s}/img/{:s}_{:d}.png".format(save_dir, prompt, seed_num).replace(" ", "_")
    
    if not vae:
        vae, tokenizer, text_encoder, unet, scheduler = _init_models()
    torch.cuda.empty_cache()
    prompt = [prompt]
    generator = torch.manual_seed(seed_num)

    # Prep text 
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8)
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0] # Need to scale to match k

    latent_at_each_step = torch.Tensor([]).to("cuda")
    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
            latent_at_each_step = torch.cat([latent_at_each_step, latents])
            
    if save_latents:
        with open(path_latents, "wb") as outfile:
            pickle.dump(latent_at_each_step, outfile)
        outfile.close()
    
    intermediate_decoded_images = []
    for step in range(num_inference_steps):
        with torch.no_grad():
            image = vae.decode(latent_at_each_step[[step]] * 1 / 0.18215)

        # Display
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0]
        
        if save_imgs:
            path_step_img = path_imgs[:-4] + "_step{:03d}.png".format(step)
            plt.imsave(path_step_img, image)
        intermediate_decoded_images.append(image)
    
    if save_final_image:
        plt.imsave(path_img, image)
    
    if save_gif:
        gif_images = intermediate_decoded_images
        for i in range(5):
            gif_images.append(image)
        vid = makeVideo(np.array(gif_images), saved=True)
        vid.save(path_gif, writer='imagemagick', fps=10)
        
    intermediate_decoded_images = np.array(intermediate_decoded_images)
        
    return {"prompt": prompt[0], "height": height, "width": width, "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale, "seed_num": seed_num, "latents": latent_at_each_step,
            "images": intermediate_decoded_images}


def generate_image(prompt, seed_num, tokenizer, text_encoder, net, vae, scheduler,
                   batch_size=1, 
                   height=512, width=512, 
                   num_inference_steps=15, 
                   guidance_scale=7.5,
                   modified_unet=None,
                   at_step=None,
                   return_latents=False,
                   stop_at_step=None,
                   ):
    torch.manual_seed(seed_num)
    # Prep text 
    text_input = tokenizer(prompt, padding="max_length", 
                           max_length=tokenizer.model_max_length, 
                           truncation=True, return_tensors="pt")

    max_length = text_input.input_ids.shape[-1]

    uncond_input = tokenizer("", padding="max_length", 
                             max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
        (batch_size, net.in_channels, height // 8, width // 8)
    )

    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0] # Need to scale to match k

    # Loop
    with autocast("cuda"):
        for j, t in enumerate(scheduler.timesteps):
            if not (stop_at_step is None) and j == stop_at_step:
                break
            sigma = scheduler.sigmas[j]
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            with torch.no_grad():
                if not (at_step is None) and j == at_step:
                    output = modified_unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                    break
                else:
                    output = net(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = output["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, j, latents)["prev_sample"]

    latents = 1 / 0.18215 * latents
    # scale and decode the image latents with vae
    if return_latents:
        return latents
    with torch.no_grad():
        decoded = vae.decode(latents)
        try:
            image = decoded["sample"]
        except:
            image = decoded
        
    return image


def makeVideo(arr, cmap=None, saved=False, title=None, interval=2000):
    '''
    makeVideo: given a 3 or 4D array (time x h x w [x 1]), returns an HTML animation 
    of array for viewing in a notebook for example. Cell could say something like:
    %%capture
    # You want to capture the output when the actual call is made. 
    vid = makeVideo(arr, cmap='gray')
    with the following cell just
    vid
    '''
    
    if len(arr.shape) == 4 and arr.shape[-1] == 1: # one channel, otherwise imshow gets confused
        arr = arr.squeeze()
        print('New arr shape {}.'.format(arr.shape))
    
    f, ax = plt.subplots(1,1, figsize=(6,6))
    dispArtist = ax.imshow(arr[0,...], interpolation=None, cmap=cmap)
    f.patch.set_alpha(0.)
    ax.axis("off")
    if title:
        ax.set_title(title)
        
    plt.tight_layout()
    
    def updateFig(i):
        # global dispArtist, arr # not sure why I don't need these:
        if i >= arr.shape[0]:
            i = 0

        dispArtist.set_array(arr[i,...])
        return (dispArtist, )
    
    ani = animation.FuncAnimation(f, updateFig, interval=interval/arr.shape[0], 
                                  frames = arr.shape[0], blit = True, repeat = False)
    
    if saved:
        ani = animation.FuncAnimation(f, updateFig, interval=interval/arr.shape[0], 
                                  frames = arr.shape[0], blit = True, repeat = True)
        return ani
    return HTML(ani.to_jshtml()) # gives a nice button interface for pause and playback.
