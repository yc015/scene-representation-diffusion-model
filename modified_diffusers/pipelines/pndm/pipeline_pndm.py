import warnings

import torch

from tqdm.auto import tqdm

from ...pipeline_utils import DiffusionPipeline


class PNDMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, num_inference_steps=50, output_type="pil", **kwargs):
        # For more information on the sampling method you can take a look at Algorithm 2 of
        # the official paper: https://arxiv.org/pdf/2202.09778.pdf

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            model_output = self.unet(image, t)["sample"]

            image = self.scheduler.step(model_output, t, image)["prev_sample"]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
