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
        fig, axis = plt.subplots()
        positions = ["x", "y", "z"]
        for hist, c in zip(self.posresidual_hists, positions):
            #print("hist.values()",hist.values())
            axis.stairs(hist.values(), hist.axes[0].edges, label=c)

        #axis.axvline(0, plt.rcParams["axes.linewidth"])
        axis.set_xlabel("Position residual (mm)")
        axis.set_ylabel("Counts")
        axis.legend()

        efig, eaxis = plt.subplots()
        e = ["Energy"]
        for hist, c in zip(self.eresidual_hists, e):
            eaxis.stairs(hist.values(), hist.axes[0].edges, label=c)

        #eaxis.axvline(0, plt.rcParams["axes.linewidth"])
        eaxis.set_xlabel("Energy residual (MeV)")
        eaxis.set_ylabel("Counts")
        eaxis.legend()
        #self.writer.add_figure(f"{self.name_prefix}/position_residuals", fig, global_step=global_step)
        wandb_logs = {
            f"{self.name_prefix}/position_residuals": wandb.Image(fig),
            f"{self.name_prefix}/energy_residuals": wandb.Image(efig)
        }

        posresidual_bias = self.posresidual_sum / self.n_points
        eresidual_bias = self.eresidual_sum / self.n_points
        for bias, c in zip(posresidual_bias, positions):
            wandb_logs[f"{self.name_prefix}/bias/{c}-mm"] = bias
            #self.writer.add_scalar(f"{self.name_prefix}/bias/{c}-mm", bias, global_step=global_step)

        #self.writer.add_scalar(f"{self.name_prefix}/bias/e-MeV", eresidual_bias.item(), global_step=global_step)
        wandb_logs[f"{self.name_prefix}/bias/e-MeV"] = eresidual_bias.item()

        posresidual_fwhm = [fwhm(hist.view(), hist.axes[0].edges) for hist in self.posresidual_hists]
        eresidual_fwhm = [fwhm(self.eresidual_hists[0].view(), self.eresidual_hists[0].axes[0].edges) ]
        for fwhm_value, c in zip(posresidual_fwhm, positions):
            #self.writer.add_scalar(f"{self.name_prefix}/fwhm/{c}-mm", fwhm_value, global_step=global_step)
            wandb_logs[f"{self.name_prefix}/fwhm/{c}-mm"] = fwhm_value
        
        
        wandb_logs[f"{self.name_prefix}/fwhm/e-MeV"] = eresidual_fwhm

        self.writer.log(wandb_logs, step=global_step)
        #self.writer.add_scalar(f"{self.name_prefix}/fwhm/e-MeV", eresidual_fwhm[0], global_step=global_step)

        plt.close(fig)
        plt.close(efig)

        r_fig, r_axis = plt.subplots(figsize=(10, 7))
        
        radial_fwhms = {}
        for (low, high), (key, hist) in zip(self.r_ranges, self.r_seg_e_hists.items()):
            label = f"{low}-{high} mm"
            r_axis.stairs(hist.values(), hist.axes[0].edges, label=label)
            
            # 5. Calculate FWHM for each range
            val = fwhm(hist.view(), hist.axes[0].edges)
            radial_fwhms[f"{self.name_prefix}/fwhm/e_{key}_mm"] = val

        r_axis.set_xlabel("Energy residual (MeV)")
        r_axis.set_ylabel("Counts")
        r_axis.set_title("Energy Residuals by Radial Distance")
        r_axis.legend()

        # 6. Log everything to WandB
        wandb_logs = {
            f"{self.name_prefix}/energy_residuals_by_radius": wandb.Image(r_fig),
            **radial_fwhms  # Merge the radial FWHM scalars into the log
        }
        
        # ... [Merge your existing position/bias logs here] ...
        
        self.writer.log(wandb_logs, step=global_step)
        plt.close(r_fig)

        if self.name_prefix == "validation_metrics":

            true_e_centers = []
            res_means = []
            res_sigmas = []

            # Iterate through our true energy slices
            for (low, high), (key, hist) in zip(self.e_true_ranges, self.e_seg_res_hists.items()):
                counts = hist.values()
                
                # Ensure we have enough statistics to make a meaningful calculation
                if np.sum(counts) > 30: 
                    bin_centers = hist.axes[0].centers
                    
                    # 1. Bias: Weighted average of the residuals
                    # This captures the "center of mass" of the distribution
                    bias = np.average(bin_centers, weights=counts)
                    
                    # 2. Uncertainty: Convert FWHM to Sigma (1-sigma error bars)
                    full_width = fwhm(counts, hist.axes[0].edges)
                    sigma = full_width / 2.355
                    
                    true_e_centers.append((low + high) / 2)
                    res_means.append(bias)
                    res_sigmas.append(sigma)

            # --- Plotting the Linearity/Resolution Curve ---
            res_fig, res_axis = plt.subplots(figsize=(8, 6))
            
            # Use errorbar to show Mean Bias +/- 1 Sigma
            res_axis.errorbar(
                true_e_centers, 
                res_means, 
                yerr=res_sigmas, 
                fmt='o', 
                color='black', 
                ecolor='blue', 
                capsize=4, 
                elinewidth=1.2,
                markerfacecolor='white',
                label=r"Mean Residual $\pm 1\sigma$"
            )
            
            # Reference line for perfect reconstruction
            res_axis.axhline(0, color='red', linestyle='--', linewidth=1, label="Ideal Recon")
            
            res_axis.set_xlabel("True Energy (MeV)")
            res_axis.set_ylabel("Energy Residual (MeV)")
            res_axis.set_title("Energy Reconstruction Bias and Resolution")
            res_axis.set_ylim(-1.5, 1.5)  # Zoom in on the bias region
            res_axis.grid(True, linestyle=':', alpha=0.6)
            res_axis.legend()

            # Final Log to WandB
            # (Make sure to combine this with your other wandb_logs dict if calling once)
            self.writer.log({
                f"{self.name_prefix}/energy_residual_vs_trueE": wandb.Image(res_fig)
            }, step=global_step)
            
            plt.close(res_fig)
