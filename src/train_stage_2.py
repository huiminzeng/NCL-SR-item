import os
import torch
from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything


def collate_fn_dp(data):
    return list(data)

def main(args):
    seed_everything(args.seed)
    if torch.cuda.is_available(): 
        args.device = 'cuda'
    else: 
        args.device = 'cpu'

    model = E5Model()
    model.init_non_cl()
    model.to(args.device)
    load_root = os.path.join('experiments/stage_1', 
                                args.dataset_code, args.ds_mode,
                                'num_replace_' + str(args.num_replace), 
                                'lambda_alignment_' + str(args.mce_gamma_a),
                                'lambda_uniformity_' + str(args.mce_gamma_u),
                                'lambda_item_' + str(args.mce_gamma_s))
    
    print("we are loading model from: ", load_root)
    model_load_name = os.path.join(load_root, 'model.checkpoint')
    if os.path.isfile(model_load_name):
        model_checkpoint = torch.load(model_load_name)
        model.load_state_dict(model_checkpoint['state_dict'])
    else:
        exit("no model found")

    train_set, val_set, _, meta = dataloader_factory_non_cl(args, model.tokenizer)
    train_set_DP, _ = get_datasets(args)

    args.export_dir = os.path.join('experiments/stage_2', 
                                args.dataset_code, args.ds_mode, 
                                'num_replace_' + str(args.num_replace), 
                                'lambda_alignment_' + str(args.mce_gamma_a),
                                'lambda_uniformity_' + str(args.mce_gamma_u),
                                'lambda_item_' + str(args.mce_gamma_s))
    
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    print("we are saving model to: ", args.export_dir)
    
    # Training
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size,
                                                shuffle=True, pin_memory=True, num_workers=args.num_workers,
                                                collate_fn=collate_fn_non_cl) 

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                            collate_fn=collate_fn_val)
    
    DP_loader = torch.utils.data.DataLoader(train_set_DP, batch_size=args.val_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                            collate_fn=collate_fn_dp)
    
    my_trainer = E5Trainer_non_cl(args=args, model=model, meta=meta, 
                                train_loader=train_loader,
                                val_loader=val_loader,
                                DP_loader=DP_loader)
    
    my_trainer.train()


if __name__ == "__main__":
    set_template(args)
    main(args)
