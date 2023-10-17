import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm import tqdm


class StableDiffusion:
    def __init__(
        self,
        vae_arch="CompVis/stable-diffusion-v1-4",
        tokenizer_arch="openai/clip-vit-large-patch14",
        encoder_arch="openai/clip-vit-large-patch14",
        unet_arch="CompVis/stable-diffusion-v1-4",
        device="cpu",
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        manual_seed=1,
    ) -> None:
        self.height = height  # default height of Stable Diffusion
        self.width = width  # default width of Stable Diffusion
        self.num_inference_steps = num_inference_steps  # Number of denoising steps
        self.guidance_scale = guidance_scale  # Scale for classifier-free guidance
        self.device = device
        self.manual_seed = manual_seed

        vae = AutoencoderKL.from_pretrained(vae_arch, subfolder="vae")
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_arch)
        text_encoder = CLIPTextModel.from_pretrained(encoder_arch)

        # The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained(unet_arch, subfolder="unet")

        # The noise scheduler
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

        # To the GPU we go!
        self.vae = vae.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.unet = unet.to(self.device)

        self.token_emb_layer = text_encoder.text_model.embeddings.token_embedding
        pos_emb_layer = text_encoder.text_model.embeddings.position_embedding
        position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
        self.position_embeddings = pos_emb_layer(position_ids)

    def get_output_embeds(self, input_embeddings):
        # CLIP's text model uses causal mask, so we prepare it here:
        bsz, seq_len = input_embeddings.shape[:2]
        causal_attention_mask = (
            self.text_encoder.text_model._build_causal_attention_mask(
                bsz, seq_len, dtype=input_embeddings.dtype
            )
        )

        # Getting the output embeddings involves calling the model with passing output_hidden_states=True
        # so that it doesn't just return the pooled final predictions:
        encoder_outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=input_embeddings,
            attention_mask=None,  # We aren't using an attention mask so that can be None
            causal_attention_mask=causal_attention_mask.to(self.device),
            output_attentions=None,
            output_hidden_states=True,  # We want the output embs not the final output
            return_dict=None,
        )

        # We're interested in the output hidden state only
        output = encoder_outputs[0]

        # There is a final layer norm we need to pass these through
        output = self.text_encoder.text_model.final_layer_norm(output)

        # And now they're ready!
        return output

    def set_timesteps(self, scheduler, num_inference_steps):
        scheduler.set_timesteps(num_inference_steps)
        scheduler.timesteps = scheduler.timesteps.to(torch.float32)

    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def generate_with_embs(self, text_embeddings, text_input, loss_fn, loss_scale):
        generator = torch.manual_seed(
            self.manual_seed
        )  # Seed generator to create the inital latent noise
        batch_size = 1

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        self.set_timesteps(self.scheduler, self.num_inference_steps)

        # Prep latents
        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
            generator=generator,
        )
        latents = latents.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        # Loop
        for i, t in tqdm(
            enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)
        ):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if i % 5 == 0:
                # Requires grad on the latents
                latents = latents.detach().requires_grad_()

                # Get the predicted x0:
                # latents_x0 = latents - sigma * noise_pred
                latents_x0 = self.scheduler.step(
                    noise_pred, t, latents
                ).pred_original_sample

                # Decode to image space
                denoised_images = (
                    self.vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
                )  # range (0, 1)

                # Calculate loss
                loss = loss_fn(denoised_images) * loss_scale

                # Occasionally print it out
                # if i % 10 == 0:
                #     print(i, "loss:", loss.item())

                # Get gradient
                cond_grad = torch.autograd.grad(loss, latents)[0]

                # Modify the latents based on this gradient
                latents = latents.detach() - cond_grad * sigma**2
                self.scheduler._step_index = self.scheduler._step_index - 1

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.latents_to_pil(latents)[0]

    def generate_image(
        self,
        prompt="A campfire (oil on canvas)",
        loss_fn=None,
        loss_scale=200,
        concept_embed=None,  # birb_embed["<birb-style>"]
    ):
        prompt += " in the style of cs"
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_input.input_ids.to(self.device)
        custom_style_token = self.tokenizer.encode("cs", add_special_tokens=False)[0]
        # Get token embeddings
        token_embeddings = self.token_emb_layer(input_ids)

        # The new embedding - our special birb word
        embed_key = list(concept_embed.keys())[0]
        replacement_token_embedding = concept_embed[embed_key]

        # Insert this into the token embeddings
        token_embeddings[
            0, torch.where(input_ids[0] == custom_style_token)
        ] = replacement_token_embedding.to(self.device)
        # token_embeddings = token_embeddings + (replacement_token_embedding * 0.9)
        # Combine with pos embs
        input_embeddings = token_embeddings + self.position_embeddings

        #  Feed through to get final output embs
        modified_output_embeddings = self.get_output_embeds(input_embeddings)

        # And generate an image with this:
        generated_image = self.generate_with_embs(
            modified_output_embeddings, text_input, loss_fn, loss_scale
        )
        return generated_image
