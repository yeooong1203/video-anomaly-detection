from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from dataset import (collate_fn_variable_length, UCFTestVideoDataset, UCFTrainVideoDataset_Stratified)
from model import Model_V2_AllCNN
from train import concatenated_train_variable_length
from utillsv2 import Concat_list_all_crop_feedback
from test import test
import option
from tqdm import tqdm
import os
import numpy as np
import wandb
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = option.parser.parse_args()
    set_seed(42)
    
    len_N, original_lables  = Concat_list_all_crop_feedback(Test=False, create='False')

    # WandB
    wandb.login()
    wandb.init(project="Unsupervised Anomaly Detection", config=args)
    
    from datetime import datetime
    import subprocess
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = wandb.run.id
    
    # Paths
    os.makedirs("unsupervised_ckpt", exist_ok=True)
    model_name = f"{args.datasetname}_{args.model_type}"
    best_path = f'unsupervised_ckpt/{model_name}_best_{ts}_{run_id}.pkl'
    final_path = f'unsupervised_ckpt/{model_name}_final_{ts}_{run_id}.pkl'
    
    test_loader = DataLoader(
        UCFTestVideoDataset("../C2FPL/Concat_test_10.npy", "list/nalist_test_i3d.npy"),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False
    )
    
    train_loader = DataLoader(
            UCFTrainVideoDataset_Stratified(
                conall_path="../C2FPL/concat_UCF.npy",
                pseudo_path=args.pseudofile,
                nalist_path="list/nalist_i3d.npy"
            ),
            batch_size=args.batch_size_video,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn_variable_length 
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model_V2_AllCNN(args.feature_size, kernel_size=args.temporal_kernel)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model Info]")
    print(f"  Type: {args.model_type}")
    print(f"  Parameters: {total_params:,}")
    

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001,                  
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 35], gamma=0.1)
    
    auc, ap = test(test_loader, model, args, device)
    print(f"\nEpoch 0 - AUC: {auc:.4f}, AP: {ap:.4f}")
    wandb.log({'AUC': auc, 'AP': ap}, step=0)
    
    best_auc = auc
    torch.save(model.state_dict(), best_path)
    print(f"Init best_auc: {best_auc:.4f} -> {best_path}")
    
    test_info = {"epoch": [], "test_auc": []}
    
    for epoch in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        loss = concatenated_train_variable_length(
                train_loader, model, optimizer, epoch, device,
                accumulation_steps=args.accumulation_steps
            )
        
        # Test
        auc, ap = test(test_loader, model, args, device)
        
        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"\n[BEST] Epoch {epoch} - AUC: {best_auc:.4f} -> {best_path}")
        
        test_info["epoch"].append(epoch)
        test_info["test_auc"].append(auc)
        
        scheduler.step()

        
        print(f'\nEpoch {epoch}/{args.max_epoch}, LR: {optimizer.param_groups[0]["lr"]:.4f}, '
              f'AUC: {auc:.4f}, AP: {ap:.4f}, Loss: {loss:.4f}\n')
        
        wandb.log({'AUC': auc, 'AP': ap, 'loss': loss}, step=epoch)
    
    # Save final
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final -> {final_path}")
    
    wandb.finish()