from abc import ABC, abstractmethod
from typing import Hashable, Iterable

import hist as h
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics as tm
import wandb
from torch import nn
from torch.utils import _pytree as pytree


from warnings import warn


def fwhm(counts: np.ndarray, bin_edges: np.ndarray) -> float:
    # 1. Validation & Setup
    if len(counts) + 1 != len(bin_edges):
        raise ValueError("len(counts) must be len(bin_edges) - 1")
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_idx = np.argmax(counts)
    half_max = counts[max_idx] / 2.0

    # 2. Split the data into Left and Right sides of the peak
    left_side_counts = counts[:max_idx + 1]
    left_side_bins = bin_centers[:max_idx + 1]
    
    right_side_counts = counts[max_idx:]
    right_side_bins = bin_centers[max_idx:]

    # 3. Find the exact X-coordinate where Y = half_max using interpolation
    # np.interp(x_target, x_data, y_data) -> finds y_target
    # We flip it: np.interp(half_max, counts, bins) -> finds exact bin position
    
    # Left side: counts must be increasing for np.interp
    left_hm_x = np.interp(half_max, left_side_counts, left_side_bins)
    
    # Right side: counts must be increasing, so we flip the arrays
    right_hm_x = np.interp(half_max, right_side_counts[::-1], right_side_bins[::-1])

    return right_hm_x - left_hm_x

class MetricMonitor(ABC):
    def __init__(self, run: wandb.Run, name_prefix: str = "validation_metrics"):
        self.run = run
        self.name_prefix = name_prefix
        self.reset()

    @abstractmethod
    def update(self, predict: dict, truth: dict):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class PositionEnergyMonitor(MetricMonitor):
    def __init__(
        self,
        run: wandb.Run,
        minpos_residual: float = -4000,
        maxpos_residual: float = 4000,
        mine_residual: float = -5,
        maxe_residual: float = 5,
        bins: int = 100,
        name_prefix: str = "validation_metrics"
    ):
        self.writer = run
        self.minpos_residual = minpos_residual
        self.maxpos_residual = maxpos_residual
        self.mine_residual = mine_residual
        self.maxe_residual = maxe_residual
        self.bins = bins
        self.r_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000)]
        self.e_true_bins = np.linspace(0.5, 10.5, 11) 
        self.e_true_ranges = [(self.e_true_bins[i], self.e_true_bins[i+1]) 
                              for i in range(len(self.e_true_bins)-1)]
        
        
        
        self.posresidual_hists = [
            h.Hist(h.axis.Regular(bins, minpos_residual, maxpos_residual, overflow=True, underflow=True, name=name))
            for name, label in zip(["x", "y", "z"], [r"$x$", r"$y$", r"$z$"])
        ]
        self.eresidual_hists = [
            h.Hist(h.axis.Regular(bins, mine_residual, maxe_residual, overflow=True, underflow=True, name="Energy", label="Energy"))
        ]
        self.r_seg_e_hists = {
            f"{low}_{high}":  h.Hist(h.axis.Regular(bins,mine_residual, maxe_residual, name=f"E_res_R_{low}_{high}")) for low,high in self.r_ranges
        }
        # Dictionary to store energy residuals indexed by true energy range
        self.e_seg_res_hists = {
            f"{low}_{high}": h.Hist(h.axis.Regular(bins, mine_residual, maxe_residual, name=f"ReconE_TrueE_{low}_{high}")) 
            for low, high in self.e_true_ranges
        }

        self.posresidual_sum = np.zeros(3, dtype=np.double)
        self.eresidual_sum = np.zeros(1, dtype=np.double)
        self.n_points: int = 0
        self.name_prefix = name_prefix


    def update(self, predict: dict[Hashable : torch.Tensor], truth: dict[Hashable : torch.Tensor]) -> None:
        #truth_positions = np.stack([truth[f"mcPos{c}"].cpu().numpy() for c in ["x", "y", "z"]], axis=-1) # (N,3)
        predict_pos = predict["position"].detach().cpu().numpy() 
        truth_pos = truth["position"].detach().cpu().numpy()
        predict_e = predict["energy"].detach().cpu().numpy()
        truth_e = truth["energy"].detach().cpu().numpy()
        residuals_pos = predict_pos - truth_pos
        residuals_e = predict_e - truth_e
        #print("residuals_e ",residuals_e)
        #print("lenresiduals_e ",len(residuals_e))
        for posres, poshist in zip(residuals_pos.T, self.posresidual_hists):
            poshist.fill(posres)
        
        self.eresidual_hists[0].fill(residuals_e)
        #print("residuals_pos.shape[0] ",residuals_pos )
        #print("residuals_e.shape[0] ",residuals_e )
        self.n_points += residuals_pos.shape[0] 
        self.posresidual_sum += residuals_pos.sum(0) # sum across x,y,z in each event
        self.eresidual_sum += residuals_e.sum(0) 

        true_radius = np.linalg.norm(truth_pos, axis = 1)
        #print("true_radius ",true_radius)
        for (low,high), hist_key in zip(self.r_ranges, self.r_seg_e_hists.keys()):
            mask = (true_radius > low) & (true_radius < high)
            if np.any(mask):
                self.r_seg_e_hists[hist_key].fill(residuals_e[mask])
        if self.name_prefix == "validation_metrics":
            # Inside update() method:
            for (low, high), hist_key in zip(self.e_true_ranges, self.e_seg_res_hists.keys()):
                mask = (truth_e >= low) & (truth_e < high)
                if np.any(mask):
                    # We store the residual (Recon - True) for this True Energy slice
                    self.e_seg_res_hists[hist_key].fill(residuals_e[mask])
        

    def reset(self) -> None:
        for hist in self.posresidual_hists:
            hist[:] = 0
        for hist in self.eresidual_hists:
            hist[:] = 0
        for hist in self.r_seg_e_hists.values():
            hist[:] = 0
        if self.name_prefix == "validation_metrics":
            for hist in self.e_seg_res_hists.values():
                hist[:] = 0
        self.residual_sum = 0
        self.n_points = 0

    def compute(self, global_step: int) -> None:
        print("In compute")
        wandb_logs = {}

        # --- 1. Position Residuals ---
        fig, axis = plt.subplots()
        positions = ["x", "y", "z"]
        for hist, c in zip(self.posresidual_hists, positions):
            axis.stairs(hist.values(), hist.axes[0].edges, label=c)
        axis.set_xlabel("Position residual (mm)")
        axis.set_ylabel("Counts")
        axis.legend()
        wandb_logs[f"{self.name_prefix}/Plots/position_residuals"] = wandb.Image(fig)
        plt.close(fig)

        # --- 2. Energy Residuals ---
        efig, eaxis = plt.subplots()
        for hist in self.eresidual_hists:
            eaxis.stairs(hist.values(), hist.axes[0].edges, label="Energy")
        eaxis.set_xlabel("Energy residual (MeV)")
        eaxis.set_ylabel("Counts")
        eaxis.legend()
        wandb_logs[f"{self.name_prefix}/Plots/energy_residuals"] = wandb.Image(efig)
        plt.close(efig)

        # --- 3. Bias & FWHM (Scalars) ---
        posresidual_bias = self.posresidual_sum / self.n_points
        eresidual_bias = self.eresidual_sum / self.n_points
        
        for bias, c in zip(posresidual_bias, positions):
            wandb_logs[f"{self.name_prefix}/bias/{c}-mm"] = float(bias)
        wandb_logs[f"{self.name_prefix}/bias/e-MeV"] = float(eresidual_bias.item())

        for hist, c in zip(self.posresidual_hists, positions):
            val = fwhm(hist.view(), hist.axes[0].edges)
            wandb_logs[f"{self.name_prefix}/fwhm/{c}-mm"] = float(val)
        
        # Fixed eresidual_fwhm to be a scalar float
        val_e = fwhm(self.eresidual_hists[0].view(), self.eresidual_hists[0].axes[0].edges)
        wandb_logs[f"{self.name_prefix}/fwhm/e-MeV"] = float(val_e)

        # --- 4. Radial Distance Plots & Radial FWHM ---
        r_fig, r_axis = plt.subplots(figsize=(10, 7))
        for (low, high), (key, hist) in zip(self.r_ranges, self.r_seg_e_hists.items()):
            label = f"{low}-{high} mm"
            r_axis.stairs(hist.values(), hist.axes[0].edges, label=label)
            
            # FIXED: Add FWHM directly to wandb_logs instead of a separate dict
            val_r = fwhm(hist.view(), hist.axes[0].edges)
            wandb_logs[f"{self.name_prefix}/fwhm/e_{key}_mm"] = float(val_r)

        r_axis.set_title("Energy Residuals by Radial Distance")
        r_axis.legend()
        wandb_logs[f"{self.name_prefix}/Plots/energy_residuals_by_radius"] = wandb.Image(r_fig)
        plt.close(r_fig)

        # --- 5. Linearity/Resolution Curve ---
        if self.name_prefix == "validation_metrics":
            true_e_centers, res_means, res_sigmas = [], [], []
            for (low, high), (key, hist) in zip(self.e_true_ranges, self.e_seg_res_hists.items()):
                counts = hist.values()
                if np.sum(counts) > 30: 
                    bin_centers = hist.axes[0].centers
                    bias = np.average(bin_centers, weights=counts)
                    full_width = fwhm(counts, hist.axes[0].edges)
                    sigma = full_width / 2.355
                    
                    true_e_centers.append((low + high) / 2)
                    res_means.append(bias)
                    res_sigmas.append(sigma)

            if true_e_centers:
                res_fig, res_axis = plt.subplots(figsize=(8, 6))
                res_axis.errorbar(true_e_centers, res_means, yerr=res_sigmas, fmt='o', color='black')
                res_axis.axhline(0, color='red', linestyle='--')
                res_axis.set_title("Energy Reconstruction Bias and Resolution")
                wandb_logs[f"{self.name_prefix}/Plots/energy_residual_vs_trueE"] = wandb.Image(res_fig)
                plt.close(res_fig)

        # --- FINAL SINGLE LOG CALL ---
        return wandb_logs
        #self.writer.log(wandb_logs, step=global_step)