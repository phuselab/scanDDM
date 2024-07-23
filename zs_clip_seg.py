import torch
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from scipy.ndimage import zoom, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt


def smooth_map(sal):
    # Saliency
    sigma = 1 / 0.039
    Z = gaussian_filter(sal, sigma=sigma)
    Z = Z / np.max(Z)
    return Z


def get_obj_map(img, text):

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    assert isinstance(processor, CLIPSegProcessor)

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    assert isinstance(model, CLIPSegForImageSegmentation)

    inputs = processor(text=text, images=[img] * len(text), return_tensors="pt")

    # predict
    with torch.no_grad():
        outputs = model(**inputs)  # type: ignore
    preds = outputs.logits.unsqueeze(1)

    salmap = np.zeros([352, 352], dtype=np.float32)
    for i in range(len(text)):
        salmap += torch.sigmoid(preds[i][0]).numpy()
    for _ in range(40):
        salmap = np.clip(salmap - salmap.mean(), 0, np.inf)
    salmap = zoom(
        salmap,
        [img.shape[0] / salmap.shape[0], img.shape[1] / salmap.shape[1]],
        order=1,
    )
    salmap = smooth_map(salmap)
    salmap /= np.max(salmap)

    return salmap
