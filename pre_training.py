import argparse
import torch
from torch.utils.data import DataLoader
import random
from torch.nn import TripletMarginLoss
import torch.optim as optim
import time
from math import inf
import wandb
from datetime import timedelta

wandb.init(project="dyslexia", entity="ailab-unifi")
random.seed(42)

from model import ResnetWrapper, ViTWrapper
from data import IAMDL


DEVICE = 'cuda:0'
XML = 'IAM/xml'
SETS = 'IAM/SETS'
DATA = 'IAM/DATA'

def train(args):

    # LOAD DATA
    train_data = IAMDL('trainset', DEVICE)
    validation_data = IAMDL('validationset', DEVICE)

    # LOAD MODEL
    # wrapper = ViTWrapper('pretrain', DEVICE, pretrain=True)
    wrapper = ResnetWrapper(name='resnet18', device=DEVICE, classes = 2, pretrain=True)
    model = wrapper.get_model()

    # TRAIN SETTINGS
    epochs = 1000
    start_epoch = 0
    batch_size = 32
    best_val_loss = inf
    exit_counter = 0
    lr = 0.0001

    loss = TripletMarginLoss(margin=1.0, p=2)
    opt = optim.AdamW(model.parameters(), lr=lr)

    if args.resume:
        start_epoch, best_val_loss, exit_counter, opt_chk = wrapper.resume('pretrain_checkpoint.pth')
        opt.load_state_dict(opt_chk)

    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    }

    print('Start Training')
    for e in range(start_epoch, epochs):

        # TRAIN
        model.train()
        loader = iter(DataLoader(train_data, batch_size=batch_size, shuffle=True))
        train_loss = 0.0
        running_loss = 0.0

        start = time.time()
        for i in range(0, int(len(train_data) / batch_size) + 1):
            samples = next(loader)
            a, p, n = train_data.batch_triplets(samples)

            opt.zero_grad()
            anchor_embed, positive_embed, negative_embed  = model(a), model(p), model(n)

            out = loss(anchor_embed, positive_embed, negative_embed)
            out.backward()
            opt.step()

            train_loss += out.item()
            running_loss += out.item()
            if i % 5 == 4:
                print(f'Epoch {e + 1} - Batch {i + 1} - Running Loss {running_loss / 5}' , end='\r')
                running_loss = 0.0
        
        train_loss = train_loss / i
        print(f'Epoch {e + 1} - Train Loss {train_loss} - Time {str(timedelta(seconds=time.time() - start))}')

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            loader = iter(DataLoader(validation_data, batch_size=batch_size, shuffle=False))
            for i in range(0, int(len(validation_data) / batch_size) + 1):
                samples = next(loader)
                a, p, n = validation_data.batch_triplets(samples)
                anchor_embed, positive_embed, negative_embed = model(a), model(p), model(n)

                out = loss(anchor_embed, positive_embed, negative_embed)
                val_loss += out.item()
            
            val_loss = val_loss / i
            print(f"Validation Loss {val_loss}")
            if val_loss < best_val_loss:
                print(f"    !- Validation improovement! {best_val_loss} -> {val_loss}")
                exit_counter = 0
                best_val_loss = val_loss
                is_best = True
            else:
                print(f"    !- No improovement!")
                is_best = False
                exit_counter += 1
            
            wandb.log({
                "train-loss": train_loss,
                "val-loss": val_loss
                })
            # Optional
            wandb.watch(model)
            
            state = {
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'exit_counter': exit_counter,
                'optimizer': opt.state_dict()
            }
            wrapper.save_state(state, is_best)
        
        if exit_counter == 20:
            print("Exit")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--resume', '-r', action="store_true",
                        help="resume training")
    train(parser.parse_args())

