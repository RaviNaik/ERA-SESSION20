import gradio as gr
import random
import torch
import pathlib

from src.utils import concept_styles, loss_fn
from src.stable_diffusion import StableDiffusion

PROJECT_PATH = "."
CONCEPT_LIBS_PATH = f"{PROJECT_PATH}/concept_libs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate(prompt, styles, gen_steps, loss_scale):
    lossless_images, lossy_images = [], []
    for style in styles:
        concept_lib_path = f"{CONCEPT_LIBS_PATH}/{concept_styles[style]}"
        concept_lib = pathlib.Path(concept_lib_path)
        concept_embed = torch.load(concept_lib)

        manual_seed = random.randint(0, 100)
        diffusion = StableDiffusion(
            device=DEVICE,
            num_inference_steps=gen_steps,
            manual_seed=manual_seed,
        )
        generated_image_lossless = diffusion.generate_image(
            prompt=prompt,
            loss_fn=loss_fn,
            loss_scale=0,
            concept_embed=concept_embed,
        )
        generated_image_lossy = diffusion.generate_image(
            prompt=prompt,
            loss_fn=loss_fn,
            loss_scale=loss_scale,
            concept_embed=concept_embed,
        )
        lossless_images.append((generated_image_lossless, style))
        lossy_images.append((generated_image_lossy, style))
    return {lossless_gallery: lossless_images, lossy_gallery: lossy_images}


with gr.Blocks() as app:
    gr.Markdown("## ERA Session20 - Stable Diffusion: Generative Art with Guidance")
    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(label="Prompt", interactive=True)
            style_selector = gr.Dropdown(
                choices=list(concept_styles.keys()),
                value=list(concept_styles.keys())[0],
                multiselect=True,
                label="Select a Concept Style",
                interactive=True,
            )
            gen_steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=30,
                step=10,
                label="Select Number of Steps",
                interactive=True,
            )

            loss_scale = gr.Slider(
                minimum=0,
                maximum=32,
                value=8,
                step=8,
                label="Select Guidance Scale",
                interactive=True,
            )

            submit_btn = gr.Button(value="Generate")

        with gr.Column():
            lossless_gallery = gr.Gallery(
                label="Generated Images without Guidance", show_label=True
            )
            lossy_gallery = gr.Gallery(
                label="Generated Images with Guidance", show_label=True
            )

        submit_btn.click(
            generate,
            inputs=[prompt_box, style_selector, gen_steps, loss_scale],
            outputs=[lossless_gallery, lossy_gallery],
        )

app.launch()
