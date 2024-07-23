import numpy as np
from race_model import RaceModel
import cv2
from scipy.stats import multivariate_t
import torch

def torch_cos(x):
	return (torch.cos(x* 0.069) + 1)/2

class race_DDM(object):
	'''
	Class implementing the multi-choice race-to-threshold model
	'''
	def __init__(self, winner, fps, downsampled_size, ffi=False, threshold=0.6, ndt=60, noise=0.1, kappa=15, phi=0.2, eta=5, device=None):
		self.device = device
		self.downsampled_size = downsampled_size
		self.num_racers = self.downsampled_size[0] * self.downsampled_size[1]
		self.patches = [(i, j) for i in range(self.downsampled_size[0]) for j in range(self.downsampled_size[1])]
		self.num_walks = 10  # number of steps to take for each video frame computation
		self.DDM_sigma = noise
		self.realizations = []
		self.winner = winner  # the racer of the current patch
		self.k = kappa  # kappa, to weight saccades length
		self.fps = fps  # Video frames per second
		self.gaze_fs = fps * self.num_walks
		self.dt = 1 / self.gaze_fs  # DDM racer dt
		self.threshold = threshold
		self.ndt = ndt  # Non Decision Time (ms)
		self.racers = RaceModel(self.num_racers, self.num_walks, self.DDM_sigma, self.ndt, self.dt, self.device)
		self.patch_residence_time = 0.0
		self.is_new_patch = True
		self.phi = phi
		self.eta = eta
		self.ffi = ffi
		self.starting_points = torch.zeros(self.num_racers, device=self.device)

	def compute_current_RDVs(self, feed_forward_inhibit):
		# Computes the Relative Decision Value for the racers
		if feed_forward_inhibit:
			# Calculate the max of the other simulations for each racer
			max_others = torch.max(torch.where(torch.eye(self.num_racers, device=self.device)[:, :, None] == 1, -float('inf'), self.DDM_simul[:, None]), dim=0)[0]
			# Compute the RDVs by subtracting max_others from DDM_simul
			RDVs = self.DDM_simul - max_others
		else:
			# Simply return a copy of the DDM_simul array
			RDVs = self.DDM_simul.clone()

		return RDVs

	def check_event_occurrence(self, RDVs):
		# Check if any racer has reached the threshold
		occur = RDVs >= self.threshold
		# Determine winners and their times
		win = occur.any(dim=1).int()
		time_win = torch.where(occur, torch.arange(occur.shape[1], dtype=torch.float32, device=self.device), float('inf'))
		time_win = time_win.min(dim=1)[0]
		return win, time_win

	def compute_values(self, prior_map):
		# Compute the values for each patch at each time instant (with video frame granularity)
		curr_prior = prior_map[self.winner]  # prior values (V_p)

		# Gazing function (Psi)
		# Computing saccades amplitude map
		hw_ratio = max(self.downsampled_size)/min(self.downsampled_size)
		if self.downsampled_size[0] > self.downsampled_size[1]:
			kx = self.k * hw_ratio
			ky = self.k 
		else:
			kx = self.k
			ky = self.k * hw_ratio
		rv = multivariate_t(self.winner, [[kx, 0], [0, ky]], df=1)  # Cauchy Distribution
		ampl_map = rv.pdf(self.patches).reshape(prior_map.shape)
		ampl_map /= np.max(ampl_map)
		ampl_map = torch.tensor(ampl_map, device=self.device)

		# Computing saccades direction map
		patches = torch.tensor(self.patches, dtype=torch.float32, device=self.device)
		diff = patches - torch.tensor(self.winner, dtype=torch.float32, device=self.device)
		directions = torch.rad2deg(torch.atan2(diff[:, 1], diff[:, 0]) + np.pi) - 180
		sac_dirs = torch_cos(directions)
		dir_map = torch.zeros_like(prior_map, dtype=torch.float32, device=self.device)
		patch_coords = patches.long()
		dir_map[patch_coords[:, 0], patch_coords[:, 1]] = sac_dirs

		# Value map
		values = prior_map * ampl_map * dir_map
		values /= torch.max(values)

		values = self.eta * values / (1 + torch.log2(1 + curr_prior))
		eps = 1e-3
		values[values <= eps] = -float("inf")

		return values  # relative values (nu_p)

	def simulate_race(self, prior_map):
		# Multi-alternative race-to-threshold model
		if self.is_new_patch:
			self.values = self.compute_values(prior_map)
			self.is_new_patch = False

		self.patch_residence_time += 1 / self.fps

		vals = self.values.reshape(-1)

		# Simulate each racer
		self.DDM_simul = self.racers.DDM_simulate(self.starting_points, vals)
		self.starting_points = self.DDM_simul[:, -1]

		RDVs = self.compute_current_RDVs(feed_forward_inhibit=self.ffi)
		self.realizations.append(RDVs)
		RDVs_a = torch.hstack(self.realizations)
		win, t_win = self.check_event_occurrence(RDVs_a)  # Check if any racer has reached the threshold

		if win.any():
			# Some racer reached the threshold
			min_t = torch.min(t_win)
			prev_fix_dur = min_t / self.gaze_fs  # previous fixation duration in sec
			prev_fix_dur = prev_fix_dur.cpu().detach().numpy()
			rdvs_at_win = RDVs_a[:, int(min_t)]

			winner = self.patches[torch.argmax(rdvs_at_win).item()]

			self.starting_points.fill_(0)  # Reset starting points
			self.realizations.clear()
			self.racers = RaceModel(self.num_racers, self.num_walks, self.DDM_sigma, self.ndt, self.dt, self.device)
		else:
			# Keep simulating in the next cycle
			winner = (None, None)
			prev_fix_dur = None

		return winner, prev_fix_dur, self.realizations#(RDVs_a, vals)


		