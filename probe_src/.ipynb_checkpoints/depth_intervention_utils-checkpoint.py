import torch
from torch import nn
from torchvision.transforms import functional
from probe_src.probe_models import probeLinearDense
from torch import autocast
from baukit import Trace, TraceDict


torch_device = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_classifiers(at_step, weights_type="", net_type="attn1_out"):
    classifier_dict = {}

    for block_type in ["down", "mid", "up"]:
        block_range_start = 0
        if block_type == "down":
            block_range = 3
            layer_range = 2
        elif block_type == "mid":
            block_range = 1
            layer_range = 1
        elif block_type == "up":
            block_range_start = 1
            block_range = 4
            layer_range = 3
        for block_ind in range(block_range_start, block_range):
            for layer_ind in range(layer_range):
                if block_type == "mid":
                    cur_layer_name = f"{block_type}_block.attentions.{layer_ind}.transformer_blocks.0.attn1.to_out.0"
                else:
                    cur_layer_name = f"{block_type}_blocks.{block_ind}.attentions.{layer_ind}.transformer_blocks.0.attn1.to_out.0"
                probe = probeLinearDense(input_dims_dict[f"{block_type}_{block_ind}"],
                                         2,
                                         scale_dict[f"{block_type}_{block_ind}"], 
                                         use_bias=False).to(torch_device)
                probe.load_state_dict(torch.load(f"probe_checkpoints/large_syn_dataset/at_step_{at_step}/segmentation_probe_{block_type}_{block_ind}_attn1_out_{layer_ind}{weights_type}.pth"))
                classifier_dict[cur_layer_name] = probe
    return classifier_dict


def prune_weight(module, threshold=0.80):
    threshold_val = torch.quantile(module.weight.abs(), threshold, dim=-1)
    with torch.no_grad():
        module.weight[0, module.weight.abs()[0] < threshold_val[0]] = 0


def make_counterfactual_label(label, angle=0, translate=[0, 0], scale=1, sheer=0, fill=0):
    label_clone = label.copy()
    label_clone = functional.affine(torch.tensor(label_clone).to(torch.long).unsqueeze(0),
                                    angle=angle, translate=translate, scale=scale, shear=sheer, fill=fill).squeeze(0)

    return label_clone


def optimize_one_inter_rep(inter_rep, layer_name, target, probe,
                           lr=1e-3, max_epoch=256, loss_func=nn.CrossEntropyLoss(), verbose=False):
    with autocast("cuda", enabled=False):
        target_clone = torch.Tensor(target).to(torch.long).to(torch_device).unsqueeze(0)

        tensor = (inter_rep.clone()).to(torch_device).requires_grad_(True)
        rep_f = lambda: tensor

        optimizer = torch.optim.Adam([tensor], 
                                     lr=lr)
        
        if verbose:
            bar = tqdm(range(max_epoch), leave=False)
        else:
            bar = range(max_epoch)
        for i in bar:
            input_tensor = rep_f()
            optimizer.zero_grad()
            probe_seg_out = probe(input_tensor)
            # Compute the loss
            loss = loss_func(probe_seg_out, target_clone)
            # Call gradient descent
            loss.backward()
            optimizer.step()
            if verbose:
                bar.set_description(f'At layer {layer_name} [{i + 1}/{max_epoch}]; Loss: {loss.item():.3f}')
        
    return rep_f().clone()


def generate_image_with_modified_internal_rep(prompt, seed_num, 
                                              tokenizer, text_encoder,
                                              unet, scheduler,
                                              vae,
                                              batch_size=1, 
                                              height=512, width=512,
                                              num_inference_steps=15,
                                              guidance_scale=7.5,
                                              modified_layer_names=[],
                                              at_steps=[],
                                              stop_at_step=None,
                                              stop_process=False,
                                              lr=None,
                                              max_epochs=None,
                                              classifier_dicts=None,
                                              cf_target=None,
                                              loss_func=None,
                                              smoothness_loss_func=None,
                                              image=None):
    torch.manual_seed(seed_num)
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
        (batch_size, unet.in_channels, height // 8, width // 8)
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
                if j in at_steps:
                    def edit_inter_rep_multi_layers_steps(output, layer_name):
                        probe = classifier_dicts[f"step_{j}"][layer_name]
                        cloned_inter_rep = output[1].unsqueeze(0).detach().clone().to(torch.float)
                        with torch.enable_grad():
                            cloned_inter_rep = optimize_one_inter_rep(cloned_inter_rep, layer_name, 
                                                                      cf_target, probe,
                                                                      lr=lr, max_epoch=max_epochs[j], 
                                                                      loss_func=loss_func)
                        output[1] = cloned_inter_rep.to(torch.float16)
                        return output
                    cur_modified_func = edit_inter_rep_multi_layers_steps
                    with TraceDict(unet, modified_layer_names, edit_output=cur_modified_func, stop=stop_process) as ret:
                        output = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                        # representation = ret.output
                else:
                    output = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = output["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, j, latents)["prev_sample"]

    latents = 1 / 0.18215 * latents
    # scale and decode the image latents with vae
    with torch.no_grad():
        try:
            image = vae.decode(latents)["sample"]
        except:
            image = vae.decode(latents)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].permute([1, 2, 0]).detach().cpu().numpy()

    return image
