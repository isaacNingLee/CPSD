import logging
import math
import os
import datasets


import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import accelerate
from accelerate.state import AcceleratorState

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.training_utils import compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from datasets import load_from_disk

# from textual_inversion.cpsd import ClipProj

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def train_ti(args, train_data_dir, class_id, class_name):

    logging_dir = os.path.join(args.output_dir, 'cpsd_logs')

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.cpsd_gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Freeze vae and text_encoder and unet
    vae.requires_grad_(False)

    unet.requires_grad_(False)


    if args.cpsd_scale_lr:
        learning_rate = (
            args.cpsd_learning_rate * args.cpsd_gradient_accumulation_steps * args.cpsd_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    # Add the placeholder token in tokenizer
    placeholder_tokens = [f'<{class_id}>']

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != 1:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )


    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(class_name, add_special_tokens=False) #################

    ### most meaningful token preprocessing ###
    token_tensor = torch.tensor(token_ids)
    full_embed = text_encoder(token_tensor.reshape(1, -1))[0]

    # compute cosine similarity between entire full_embed and each of its component
    with torch.no_grad():
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(full_embed, full_embed.reshape(len(token_ids), 1, 768)).mean(dim=1)
        best_idx = cos_sim.argmax().item()


    #######

    initializer_token_id = token_ids[best_idx] # take first one, TODO: how to deal with multi ????(like komodo dragon, komodo or dragon???), most meaningful token preprocessing
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone() ###[CLS] token



    optimizer = optimizer_cls(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
        betas=(args.cpsd_adam_beta1, args.cpsd_adam_beta2),
        weight_decay=args.cpsd_adam_weight_decay,
        eps=args.cpsd_adam_epsilon,
    )

    dataset = load_from_disk(train_data_dir)
    column_names = dataset.column_names


    image_column = args.image_column
    caption_column = args.caption_column
    desc_column = args.desc_column

    if image_column not in column_names:
        raise ValueError(f"Image column {image_column} not found in the dataset.")
    if caption_column not in column_names:
        raise ValueError(f"Caption column {caption_column} not found in the dataset.")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = []

        if args.use_blip:
            for _, desc in zip(examples[caption_column], examples[desc_column]):
                captions.append(f"{placeholder_tokens[0]}, {desc}")
        else:
            for _ in examples[caption_column]:

                captions.append(placeholder_tokens[0])

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.cpsd_resolution, antialias=True),
            transforms.RandomResizedCrop(args.cpsd_resolution, scale=(0.6, 1.0), interpolation=PIL_INTERPOLATION["lanczos"]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.cpsd_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.cpsd_gradient_accumulation_steps)


    lr_scheduler = get_scheduler(
        args.cpsd_lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.cpsd_lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_update_steps_per_epoch * args.cpsd_num_train_epochs,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )


    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
       

    # Move text_encode and vae to gpu and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)



    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers('cpsd', tracker_config)


    # Train!
    total_batch_size = args.cpsd_batch_size * accelerator.num_processes * args.cpsd_gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.cpsd_num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.cpsd_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.cpsd_gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.cpsd_num_train_epochs * num_update_steps_per_epoch}")
    global_step = 0
    first_epoch = 0


    initial_global_step = 0


    progress_bar = tqdm(
        range(0, args.cpsd_num_train_epochs * num_update_steps_per_epoch),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    text_encoder.train()
    for epoch in range(first_epoch, args.cpsd_num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()


                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0].to(dtype=weight_dtype)


                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                


                if args.cpsd_snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.cpsd_snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                
                dist_loss = 0
                if args.cpsd_dist_match:
                    mse_loss_weights = mse_loss_weights.view(bsz, 1, 1, 1)
                    model_pred_ws = (model_pred.float() * mse_loss_weights).sum(dim=0)
                    target_ws = (target.float() * mse_loss_weights).sum(dim=0)
                    dist_loss = F.mse_loss(model_pred_ws, target_ws, reduction="mean") 
                    loss = loss + dist_loss * args.cpsd_dist_match


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.cpsd_batch_size)).mean()
                train_loss += avg_loss.item() / args.cpsd_gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), args.cpsd_max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0


            logs = {"step_loss": loss.detach().item(), "dist_loss": dist_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if args.cpsd_max_steps is not None and step >= args.cpsd_max_steps:
                break


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.output_dir + f'/ti_embeddings' is not None:
            os.makedirs(args.output_dir + f'/ti_embeddings', exist_ok=True)

        weight_name = f"class_{class_id}.safetensors"
        save_path = os.path.join(args.output_dir + f'/ti_embeddings', weight_name)
        save_progress(
            text_encoder,
            placeholder_tokens,
            placeholder_token_ids,
            accelerator,
            save_path,
            safe_serialization=True,
        )


    accelerator.end_training()

def save_progress(text_encoder, placeholder_tokens, placeholder_token_ids, accelerator, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {placeholder_tokens[0]: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)