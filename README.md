# ERA-SESSION20 - Stable Diffusion: Generative Art with Guidance
HuggingFace Space Link: https://huggingface.co/spaces/RaviNaik/ERA-SESSION20

### Tasks:
1. :heavy_check_mark: Select 5 different styles of your choice and show output for the same prompt using these 5 different styles. Remember the seeds as you'll use them later. Keep seeds different for each 5 types.
2. :heavy_check_mark: Now apply your own variant of "blue_loss" (it cannot be red, green, or blue loss) on the same prompts with each concept library and store the results.
3. :heavy_check_mark: Convert this into an app on Huggingface (and share on LinkedIn to get 500 extra points)
4. Share:
 - :heavy_check_mark: the link to your post
 - :heavy_check_mark: share the GitHub link where I can find your colab file and creative loss details.

**Styles Used:**
1. [Oil style](https://huggingface.co/sd-concepts-library/oil-style)
2. [Xyz](https://huggingface.co/sd-concepts-library/xyz)
3. [Allante](https://huggingface.co/sd-concepts-library/style-of-marc-allante)
4. [Moebius](https://huggingface.co/sd-concepts-library/moebius)
5. [Polygons](https://huggingface.co/sd-concepts-library/low-poly-hd-logos-icons)

### HuggingFace Space UI:
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/a8076e33-9d89-4a78-84b7-ad9569873560)


### Result of Experiments with different styles:
**Prompt:** `"a cat and dog in the style of cs"` \
_"cs" in the prompt refers to "custom style" whose embedding is replaced by each of the concept embeddings shown below_
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/1effe375-6ef4-4adc-be7b-d6311fdaa50d)

---
**Prompt:** `"dolphin swimming on Mars in the style of cs"`
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/2cd32248-4233-42c0-97c0-00e1ae8fdc85)

### Result of Experiments with Guidance loss functions:
**Prompt:** `"a mouse in the style of cs"`
**Loss Function:**
```python
def loss_fn(images):
    return images.mean()
```
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/c9d46e14-44bb-4ea7-88a4-26ef46344fce)
---
```python
def loss_fn(images):
    return -images.median()/3
```
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/2649e4f6-3de5-4e54-8f22-3d65874b7b07)
---
```python
def loss_fn(images):
    error = (images - images.min()) / 255*(images.max() - images.min())
    return error.mean()
```
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/6399c780-e9b7-42f8-8d90-44c8b40d5265)
---
**Prompt:** `"angry german shephard in the style of cs"`
```python
def loss_fn(images):
    error1 = torch.abs(images[:, 0] - 0.9)
    error2 = torch.abs(images[:, 1] - 0.9)
    error3 = torch.abs(images[:, 2] - 0.9)
    return (
        torch.sin(error1.mean()) + torch.sin(error2.mean()) + torch.sin(error3.mean())
    ) / 3
```
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/fa7d30ed-4efd-4504-b89c-94e093f51f9c)

---
**Prompt:** `"A campfire (oil on canvas)"`
```python
def loss_fn(images):
    error1 = torch.abs(images[:, 0] - 0.9)
    error2 = torch.abs(images[:, 1] - 0.9)
    error3 = torch.abs(images[:, 2] - 0.9)
    return (
        torch.sin((error1 * error2 * error3)).mean()
        + torch.cos((error1 * error2 * error3)).mean()
    )
```
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/88382dae-6701-4103-a664-ed17727b690f)

---
```python
def loss_fn(images):
    error1 = torch.abs(images[:, 0] - 0.9)
    error2 = torch.abs(images[:, 1] - 0.9)
    error3 = torch.abs(images[:, 2] - 0.9)
    return (
        torch.sin(error1.mean()) + torch.sin(error2.mean()) + torch.sin(error3.mean())
    ) / 3
```
![image](https://github.com/RaviNaik/ERA-SESSION20/assets/23289802/0ab3edad-579d-4821-b992-6c18b61bd444)


