import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def adapter(embedding, num_samples=100, variance=0.1):
    # embedding: [1, D]
    D = embedding.shape[-1]
    noise = torch.randn((num_samples, D), device=embedding.device) * variance
    samples = embedding + noise
    samples = samples / samples.norm(dim=-1, keepdim=True)  # normalize
    return samples


# Load image & text
image = Image.open(r"C:\Users\ander\Downloads\KAND\rise_test\Rusty_Blackbird_0015_6885.jpg").convert("RGB")
image = image.resize((224, 224))  # Ensure the image fills the expected CLIP input size
text = clip.tokenize(["a photo of a bird"]).to(device)

# Preprocess
image_tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# Encode text once
with torch.no_grad():
    text_embedding = model.encode_text(text)  # [1, D]
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

# ---- RISEWithAdapter Saliency Computation ----
# Be sure to define or load your `adapter` before this block.
from RISEWithAdapter import RISEWithAdapter
from RISE import RISE  # To generate masks

# Generate RISE masks
input_size = (224, 224)
rise = RISE(model, input_size)
rise.generate_masks(N=100, s=8, p1=0.5)
masks = rise.masks  # Use the generated masks

# Assume `adapter` is your trained adapter object already loaded and ready
# You need to define or import your adapter here before using it below

# Initialize RISEWithAdapter
explainer = RISEWithAdapter(
    clip_model=model,
    adapter=adapter,
    text_embedding=text_embedding,
    input_size=input_size,
    gpu_batch=50,
    num_samples=100
)
explainer.set_masks(masks)

# Compute saliency
with torch.no_grad():
    saliency_map = explainer(image_tensor)

saliency_np = saliency_map.cpu().numpy()
# import torch.nn.functional as F
# saliency_softmax = F.softmax(saliency_map.view(-1), dim=0).view(224, 224).cpu().numpy()

# saliency_np = saliency_map.cpu().numpy()
# saliency_vis = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min())
saliency_np -= saliency_np.min()

print("Min:", saliency_map.min().item())
print("Max:", saliency_map.max().item())
print("Mean:", saliency_map.mean().item())
print("Std Dev:", saliency_map.std().item())


# Visualize the result
plt.imshow(image)
im = plt.imshow(saliency_np, cmap='jet', alpha=0.5)
plt.colorbar(im, label = 'Saliancy value')
plt.axis('off')
plt.title("Saliency Map")
plt.show()