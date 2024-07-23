import numpy as np
import matplotlib.pyplot as plt
from scanDDM import scanDDM
from vis import draw_scanpath, compute_density_image, get_saccade_stats
import seaborn as sns
import cv2
sns.set_context("talk")


# Data path ----------------------------------------------------------
#img_path = "data/smiley.jpg"
img_path = "data/buttercat.jpg"


# Load image ---------------------------------------------------------
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
typical_shape = (768, 1024, 3)

if img.shape != typical_shape:
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(img, (768, 1024))
    else:
        img = cv2.resize(img, (1024, 768))


# Experiment Parameters ----------------------------------------------
fps = 25            #Frames per seconf
exp_dur = 2.        #Experiment duration (seconds)
n_obs = 100         #number of observers (scanpaths) to simulate


# Model Parameters ---------------------------------------------------
k = 10                  #Cauchy distribution dispersion
threshold = 1.0         #Race Model threshold
noise = 7               #Race Model diffusion strenght
eta = 17                #Race Model baseline accumulation

prompt = ["delicate"]
    

# Model Definition ----------------------------------------------------
model = scanDDM(
    experiment_dur=exp_dur,
    fps=fps,
    threshold=threshold,
    noise=noise,
    kappa=k,
    eta=eta,
    device="cpu",
)

# Simulate ------------------------------------------------------------
scans, prior_map = model.simulate_scanpaths(
    image=img, prompt=prompt, n_observers=n_obs
)
all_scans = np.vstack(scans)
prompt = ", ".join(prompt)

# plot ----------------------------------------------------------------
sp_to_plot = 1          #idx of the simulated scanpath to plot

fig = plt.figure(tight_layout=True, figsize=(15,10))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Original image")

plt.subplot(1, 3, 2)
plt.imshow(img)
draw_scanpath(
    scans[sp_to_plot][:, 0], scans[sp_to_plot][:, 1], scans[sp_to_plot][:, 2] * 1000
)
plt.axis("off")
plt.title("Simulated Scan")

plt.subplot(1, 3, 3)
sal = compute_density_image(all_scans[:, :2], img.shape[:2])
res = np.multiply(img, np.repeat(sal[:,:,None]/np.max(sal),3, axis=2))
res = res/np.max(res)
plt.imshow(res)
plt.axis("off")
plt.title("Generated Saliency ("+str(n_obs)+" scanpaths)")

fig.suptitle(prompt, fontsize=20)

plt.show()
