import numpy as np
from pixel_race_mcDDM import race_DDM
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from zs_clip_seg import get_obj_map
import torchvision.transforms as T
import torch
import warnings
warnings.filterwarnings("ignore")

class scanDDM(object):
	def __init__(self, experiment_dur, fps, task_driven=True, ffi=True, threshold=1., ndt=5, noise=0.1, kappa=10, eta=7, device=None):
		self.DDM_sigma = noise		
		self.k = kappa											#kappa, to weight saccades lenght
		self.fps = fps											#Video frames per second
		self.threshold = threshold
		self.ndt = ndt 											#Non Decision Time (ms)
		self.eta = eta
		self.ffi = ffi
		self.exp_dur = experiment_dur
		self.task_driven = task_driven
		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

	def simulate_scanpaths(self, n_observers, image=None, saliency_map=None, prompt=None):
		scans = []

		if saliency_map is None:
			assert image is not None; "Please provide an image or a precomputed saliency map"
			
			#Task driven attention
			assert prompt is not None; "Please provide an item to search as a prompt"
			#Computing Task oriented Saliency
			obj_map = get_obj_map(image, prompt)
			obj_map = torch.tensor(obj_map[None,None,:,:], device=self.device)
			saliency_map = obj_map

		reshaped_saliency = T.Resize(size=15)(saliency_map).squeeze()
		reshaped_saliency = (reshaped_saliency / torch.max(reshaped_saliency) + torch.finfo(torch.float32).eps)

		img_size = saliency_map.squeeze().shape
		nFrames = int(self.fps*self.exp_dur)
		dwnsampled_size = reshaped_saliency.shape
		ratio = np.array(img_size[:2]) / dwnsampled_size
		cov = [[0.5, 0], [0, 0.5]]  # diagonal covariance

		#Simulating Scanpaths
		for s in range(n_observers):
			mean = (dwnsampled_size[0]//2, dwnsampled_size[1]//2)
			x, y = np.random.multivariate_normal(mean, cov, 1).T			
			curr_fix = (int(x),int(y))
			prev_fix = (None, None)
			scan = [curr_fix]
			durs = []
			for iframe in range(nFrames):
				if curr_fix != (None,None):
					if prev_fix != curr_fix:
						race_model = race_DDM(winner=curr_fix, fps=self.fps, downsampled_size=dwnsampled_size, threshold=self.threshold, noise=self.DDM_sigma, kappa=self.k, eta=self.eta, ffi=self.ffi, device=self.device)
				curr_fix, prev_fix_dur, rls  = race_model.simulate_race(reshaped_saliency)

				if curr_fix != (None,None):
					scan.append(np.array(curr_fix))
					durs.append(prev_fix_dur)

			scan_np = np.vstack(scan)[:-1]
			scan_np = np.flip(scan_np,1)
			durs_np = np.array(durs)
			scan_np[:,0] = scan_np[:,0] * ratio[0] + np.random.normal(20,10,len(scan_np))
			scan_np[:,1] = scan_np[:,1] * ratio[1] + np.random.normal(20,10,len(scan_np))
			scan_dur = np.hstack([scan_np, durs_np[:,None]])
			scans.append(scan_dur)

		return scans, saliency_map.cpu().detach().numpy().squeeze()