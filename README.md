# Beyond Surface Statistics: Scene Representations in a Latent Diffusion Model
Linear probes found controllable representations of scene attributes in a text-to-image diffusion model

Project page of "Beyond Surface Statistics: Scene Representations in a Latent Diffusion Model"  
Paper arXiv link: [https://arxiv.org/abs/2306.05720](https://arxiv.org/abs/2306.05720)  
[[NeurIPS link]](https://nips.cc/virtual/2023/74894)  [[Poster link]](https://nips.cc/media/PosterPDFs/NeurIPS%202023/74894.png?t=1701540884.728899)  


## How to generate a short video of moving foreground object using a pretrained text-to-image generative model?
See [application_of_intervention.ipynb](https://github.com/yc015/scene-representation-diffusion-model/blob/main/application_of_intervention.ipynb) for how to use our intervention technique to generate a short video of moving objects.

### Some examples:

<div style="display:flex; flex-wrap:wrap; padding: 0; margin: 0;">
  <img src="https://github.com/yc015/scene-representation-diffusion-model/blob/main/resources/southern_container_plants.gif" width="300px" padding="0" margin="0"/>
  <img src="https://github.com/yc015/scene-representation-diffusion-model/blob/main/resources/macy_handbag.gif" width="300px" padding="0" margin="0"/>
</div>

The gifs are sampled using the original text-to-image diffusion model without fine-tuning. All frames are generated using the **same prompt, random seed (inital latent vectors), and model**. We edited the intermediate activations of the latent diffusion model when it generated the images so its internal representtaion of foreground match with our reference mask. See [notebook](https://github.com/yc015/scene-representation-diffusion-model/blob/main/application_of_intervention.ipynb) for implementation details.

![](https://github.com/yc015/scene-representation-diffusion-model.github.io/blob/main/resources/application_of_intervention.png)


## Probe Weights:
Unzip the probe_checkpoints.zip to acquire all probe weights trained by us. The probe weights in the unzipped folder should be sufficient for you to run all experiments shown in our paper. 


## Citation
If you find the source code of this repo helpful, please cite  

    @article{chen2023beyond,
      title={Beyond Surface Statistics: Scene Representations in a Latent Diffusion Model},
      author={Chen, Yida and Vi{\'e}gas, Fernanda and Wattenberg, Martin},
      journal={arXiv preprint arXiv:2306.05720},
      year={2023}
    }
