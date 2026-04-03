import json
import jinja2
import importlib
import torch
import wandb
from pathlib import Path
from loops import train
from utils.datasets import collate_varlen
import os

def get_class(class_path: str):
    """Converts 'utils.datasets.PosEnergyRecoDataset' into the actual Class."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_config(json_path):
    with open(json_path, 'r') as f:
        # Direct load, no templates, no rendering
        return json.load(f)

def main():
    wandb.login()
    config = load_config("/home/huangp/deeprecon/example_config/train.json")
    #config = load_config("/home/huangp/deeprecon/example_config/test.json")
    t_cfg = config['train']
    os.makedirs(t_cfg["wandbcachepath"],exist_ok=True)
    run_obj = wandb.init(project=t_cfg["project"], name= t_cfg["name"],dir=t_cfg["wandbcachepath"],entity=t_cfg['entity'], tags=t_cfg['tags'])
    #train_data_path = config['data']['train_files']
    #val_data_path   = config['data']['val_files']
    #pmt_info_path   = config['data']['pmt_valid_file']
    #ckpt_output     = config['train']['checkpoint_dir']
    
    # 2. Manually Instantiate the Model
    model_cls = get_class(config['model']['class_path'])
    model = model_cls(**config['model']['init_args'])

    # 3. Manually Instantiate the Dataloaders
    # We create the Dataset first, then wrap it in the DataLoader
    def make_loader(loader_cfg):
        ds_cfg = loader_cfg['init_args']['dataset']
        ds_cls = get_class(ds_cfg['class_path'])
        dataset = ds_cls(**ds_cfg['init_args'])
        
        collate_map = {"collate_varlen": collate_varlen}
        collate_fn = collate_map.get(loader_cfg['collate_fn_name'])
        # Pull out DataLoader args (batch_size, num_workers, etc.)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=loader_cfg['init_args']['batch_size'],
            num_workers=loader_cfg['init_args']['num_workers'],
            collate_fn=collate_fn
        )

    train_loader = make_loader(t_cfg['train_dataloader'])
    val_loader = make_loader(t_cfg['val_dataloader'])

    # 4. Instantiate optimiser and Scheduler
    #optim_cls = get_class(t_cfg['optimiser']['class_path'])
    #optimiser = optim_cls(model.parameters(), **t_cfg['optimiser']['init_args'])
    optim_cls = get_class(t_cfg['optimiser']['class_path'])
    init_args = t_cfg['optimiser']['init_args'].copy()

    wd_value = init_args.pop('weight_decay', 0.0)

    decay_params = []
    no_decay_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Exclude biases and LayerNorm parameters (weights and biases) from decay
        if "bias" in n or "norm" in n:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    # 4. Create the parameter group list
    optim_groups = [
        {"params": decay_params, "weight_decay": wd_value},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    # 5. Initialize with groups instead of model.parameters()
    optimiser = optim_cls(optim_groups, **init_args)

    
    sched_cls = get_class(t_cfg['scheduler']['class_path'])
    scheduler = sched_cls(optimiser, **t_cfg['scheduler']['init_args'])

    # 5. Instantiate Loss and Metrics
    loss_fn = get_class(t_cfg['loss_fn']['class_path'])()
    val_metric_args = t_cfg['val_metric']['init_args']
    inner_path = val_metric_args['metric_fn']['class_path']
    metric_fn_obj = get_class(inner_path)()
    val_metric_cls = get_class(t_cfg['val_metric']['class_path'])
    val_metric = val_metric_cls(metric_fn=metric_fn_obj)
    monitor_cfg = t_cfg['metric_monitor']
    trainmetric_monitor = get_class(monitor_cfg['class_path'])(run=run_obj)
    valmetric_monitor = get_class(monitor_cfg['class_path'])(run=run_obj)    
    # 6. Setup WandB
    #run = wandb.init(project=t_cfg['project'], entity=t_cfg['entity'], tags=t_cfg['tags'])

    # 7. Start Training
    train(
        checkpoint_dir=t_cfg['checkpoint_dir'],
        run=run_obj,
        log_interval=200, # or from config
        model=model,
        device=t_cfg['device'],
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimiser=optimiser,
        loss_fn=loss_fn,
        val_metric=val_metric,
        val_num_steps=t_cfg['val_num_steps'],
        num_steps=t_cfg['num_steps'],
        max_grad_norm=t_cfg['max_grad_norm'],
        scheduler=scheduler,
        trainmetric_monitor = trainmetric_monitor,
        valmetric_monitor = valmetric_monitor
    )

if __name__ == "__main__":
    main()