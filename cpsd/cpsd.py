from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from typing import List, Optional, Tuple, Union


class CPSDPipeline(StableDiffusionPipeline):


    def load_embed_trans(self, embed_trans_path: str):
        embed_trans = torch.load(embed_trans_path).to(torch.float16)
        self.embed_trans = embed_trans.to(self.device)

    def unload_embed_trans(self):

        self.embed_trans = None

    def encode_prompt(self, *args, **kwargs):

        prompt_embeds, negative_prompt_embeds = super().encode_prompt(*args, **kwargs)

        if self.embed_trans is not None:
            prompt_embeds = self.embed_trans(prompt_embeds.reshape(-1, 77)).reshape(prompt_embeds.shape)

        return prompt_embeds, negative_prompt_embeds

    def boomerang_aug(self, img, *args, **kwargs):

        self.embed_trans = None

        latents = self.encode_latents(img)
        # rnd percent noise between 0.02 and 0.5
        rnd_percent_noise = torch.rand(1).item() * 0.18 + 0.02
        z = self.boomerang_forward(rnd_percent_noise, latents)

        aug_img = self.boomerang_reverse(latents=z, percent_noise=rnd_percent_noise, *args, **kwargs)    

        return aug_img


    def encode_latents(self, img: torch.Tensor):
        with torch.no_grad():
            # Project image into the latent space
            clean_z = self.vae.encode(img).latent_dist.mode() # From huggingface/diffusers/blob/main/src/diffusers/models/vae.py
            clean_z = self.vae.config.scaling_factor * clean_z

        return clean_z



    def boomerang_forward(self, percent_noise, latents):
        '''
        Add noise to the latents via the pipe noise scheduler, according to percent_noise.
        '''

        assert percent_noise <= 0.999
        assert percent_noise >= 0.02

        # Add noise to the latent variable
        # (this is the forward diffusion process)
        noise = torch.randn(latents.shape).to(self.device)
        timestep = torch.Tensor([int(self.scheduler.config.num_train_timesteps * percent_noise)]).to(self.device).long()
        z = self.scheduler.add_noise(latents, noise, timestep).half()

        return z



    def boomerang_reverse(self, prompt, percent_noise, latents, num_inference_steps=10):
        '''
        Denoise the noisy latents according to percent_noise.
        '''

        assert percent_noise <= 0.999
        assert percent_noise >= 0.02

        # Run the reverse boomerang process
        with autocast('cuda'):
            return self(prompt=prompt, percent_noise=percent_noise, latents=latents,
                        num_inference_steps=num_inference_steps, output_type='pt').images.float()



    


class CPSDPlusPipeline(CPSDPipeline):

    def load_embed_trans(self, embed_trans_path: str):

        loaded_dict = torch.load(embed_trans_path)

        self.embed_trans = loaded_dict['embed_trans'].to(torch.float16)
        self.latent_trans = loaded_dict['latent_trans'].to(torch.float16)

    def decode_latents(self, latents):


        latents = 1 / self.vae.config.scaling_factor * latents
        latents = self.latent_trans(latents.reshape(-1, self.latent_trans.in_features)).reshape(latents.shape)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

class CPSDContPipeline(CPSDPipeline):

    def load_embed_trans(self, embed_trans_path: str):

        loaded_dict = torch.load(embed_trans_path)
        self.cpsd_mean = loaded_dict['cpsd_mean']
        self.cpsd_std = loaded_dict['cpsd_std']

        self.embed_trans = loaded_dict['embed']

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:

            latents = self.randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            if self.cpsd_mean is not None:
                latents = latents * self.cpsd_std + self.cpsd_mean
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def randn_tensor(
        self,
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
    ):
        """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # device on which tensor is created defaults to device
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cpu":
                rand_device = "cpu"
                # if device != "mps":
                #     logger.info(
                #         f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                #         f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                #         f" slighly speed up this function by passing a generator that was created on the {device} device."
                #     )
            elif gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents