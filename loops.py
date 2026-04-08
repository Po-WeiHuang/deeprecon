import math
from pathlib import Path
from typing import Iterable

import torch
import tqdm
import uproot
import wandb
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils import data

from metrics.metric_monitor import MetricMonitor

TQDM_KWARGS = {
    "bar_format": "{desc:<10} {percentage:>5.1f}% |[{bar}]{r_bar}",
    "ascii": " =",
    "unit": "batch",
    "dynamic_ncols": True,
}

def to_device(d: dict, device: str | torch.device) -> dict:
    # remove the tensor to device if tensor, if not then leave it
    def to(x):
        try:
            return x.to(device)
        except AttributeError:
            return x
    return  pytree.tree_map(lambda x: to(x), d)

def detach_to_cpu(d: dict) -> dict:
    def to(x):
        try:
            return x.detach().cpu()
        except AttributeError:
            return x
    return pytree.tree_map(lambda x: to(x), d)

@torch.inference_mode()
def validate(
    dataloader: data.DataLoader,
    run: wandb.Run,
    device: str | torch.device,
    model: nn.Module,
    metric:object,
    loss_fn: nn.Module | None = None , 
    global_step: int | None = None,
    metric_monitor: MetricMonitor | None = None
) -> float:
    model.to(device)
    model.eval()

    if metric_monitor is not None:
        if global_step is None:
            raise ValueError("'global_step' must be given a value that is not 'None'")
        metric_monitor.reset()
    metric.reset()

    step = 0
    rolling_loss = 0
    rolling_posloss = 0
    rolling_eloss = 0
    rolling_evtloss = 0

    for inputs, truth, cu_seq, max_len in tqdm.tqdm(dataloader):
        inputs = to_device(inputs, device)
        truth = to_device(truth, device)
        cu_seq = to_device(cu_seq, device)
        max_len= to_device(max_len, device)
        predict = model(**inputs,cu_seq=cu_seq, max_len=max_len)
        if loss_fn is not None:
            norm_truth = model.output_normalise(truth)
            loss, posloss, eloss, evtloss = loss_fn(predict, norm_truth)
            rolling_loss += loss.item()
            rolling_posloss += posloss.item()
            rolling_eloss += eloss.item()
            rolling_evtloss += evtloss.item()

        predict = model.output_unnormalise(predict)
        metric.update(predict, truth)
        

        if metric_monitor is not  None:
            metric_monitor.update(predict=predict, truth=truth)
        step += 1
        
    val_wandb_data = {}
    if loss_fn is not None:
        val_wandb_data.update({
           "Loss/valtotloss": rolling_loss / step,
            "Loss/valposloss": rolling_posloss / step,
            "Loss/valeloss": rolling_eloss / step,
            "Loss/valevtloss": rolling_evtloss / step 
        })
        
        

        
    if metric_monitor is not None:
        monitor_data = metric_monitor.compute(global_step)
        val_wandb_data.update(monitor_data)
    run.log(val_wandb_data, step=global_step)

    return metric.compute()


def train(
    checkpoint_dir: str | Path,
    run: wandb.Run,
    log_interval: int,
    model: nn.Module,
    device: str | torch.device,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    val_metric: object,
    val_num_steps: int,
    val_metric_is_inverted: bool = True,
    num_steps: int | None = None,
    max_grad_norm: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,# learning rate scheduling
    trainmetric_monitor: MetricMonitor | None = None,
    valmetric_monitor: MetricMonitor | None = None,
    autocast_dtype: torch.dtype = torch.bfloat16
):
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True,parents=True)

    model.to(device)
    model.train()
    #loss_fn.to(device)

    sub_epoch = 0
    step_num = 0
    rolling_loss = 0.0
    rolling_posloss = 0.0
    rolling_eloss = 0.0
    rolling_evtloss = 0.0
    dset_size = 0
    first_dset_print = True 
    training = True
    trainmetric_monitor.name_prefix = "train_metrics"
    valmetric_monitor.name_prefix = "validation_metrics"
    trainmetric_monitor = None

    with tqdm.tqdm(desc="Train", total=num_steps, **TQDM_KWARGS) as progress_bar:
        while training:
            for i, (inputs, truth, cu_seq, max_len) in enumerate(train_dataloader):
                dset_size += len(next(iter(inputs.values())))
                step_num += 1

                if num_steps is not None and step_num == num_steps:
                    training = False
                    break
                log_this_step = step_num % log_interval == 0
                #print("log_this_step",log_this_step)
                progress_bar.update()
                optimiser.zero_grad()
                inputs = to_device(inputs, device)
                truth = to_device(truth, device)
                cu_seq = to_device(cu_seq, device)
                max_len = to_device(max_len, device)
                #with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                truth = model.output_normalise(truth)
                predict = model(**inputs,cu_seq=cu_seq, max_len=max_len)
                

                loss, posloss, eloss, evtloss = loss_fn(predict, truth)
                rolling_loss += loss.item()
                rolling_posloss += posloss.item()
                rolling_eloss += eloss.item()
                rolling_evtloss += evtloss.item()
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite")
                if log_this_step:
                    train_logs = {
                        "Loss/traintotloss": rolling_loss / log_interval,
                        "Loss/trainposloss": rolling_posloss / log_interval,
                        "Loss/traineloss": rolling_eloss / log_interval,
                        "Loss/trainevtloss": rolling_evtloss / log_interval,
                    }
                    rolling_loss = 0.0
                    rolling_posloss = 0.0
                    rolling_eloss = 0.0
                    rolling_evtloss = 0.0

                    


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)
                optimiser.step()


                if scheduler is not None:
                    scheduler.step()
                    if log_this_step:
                        train_logs["learning_rate"] = scheduler.get_last_lr()[0]
                        #run.log({"learning_rate": scheduler.get_last_lr()[0]}, step=step_num)
                # look at the metrics during train process
                if log_this_step:
                    predict = model.output_unnormalise(predict)
                    truth = model.output_unnormalise(truth)

                    if trainmetric_monitor is not  None:
                        trainmetric_monitor.update(predict=predict, truth=truth)
                        monitor_data = trainmetric_monitor.compute(step_num)
                        train_logs.update(monitor_data)
                        trainmetric_monitor.reset()

                    run.log(train_logs, step=step_num)

                if step_num % val_num_steps == 0:
                    del inputs, truth, predict,loss
                    log_this_step = True
                    val_loss = validate(
                        val_dataloader,
                        run,
                        device = device,
                        model=model,
                        metric=val_metric,
                        loss_fn=loss_fn,
                        global_step=step_num,
                        metric_monitor=valmetric_monitor,
                    )
                    model.train()

                    if val_metric_is_inverted:
                        val_loss = {key: -value for key, value in val_loss.items()}
                    state_dict = {
                        "sub_epoch": sub_epoch,
                        "model": detach_to_cpu(model),
                        "optimiser": detach_to_cpu(optimiser.state_dict()),
                        "scheduler": None if scheduler is None else detach_to_cpu(scheduler.state_dict()),
                    }


                    filename = f"sub_epoch={sub_epoch}_val_loss={val_loss['total']}.pt"
                    torch.save(state_dict, checkpoint_dir/filename)

                    sub_epoch += 1
                
                #if log_this_step:
                #    run.log({},step=step_num, commit=True)
            if first_dset_print:
                print(f"Dataset size: {dset_size}")
                first_dset_print = False

    print("Training completed")


        


