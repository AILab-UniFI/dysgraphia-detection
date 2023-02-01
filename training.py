import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import shap
import torch
from torch.utils.data import DataLoader
import random
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import time
from math import inf
import wandb
from datetime import timedelta

wandb.init(project="dyslexia", entity="ailab-unifi")
random.seed(42)

from model import ViTWrapper, ResnetWrapper
from data import DisgraphiaDL
from path import *

def train(args):

    # LOAD DATA
    if args.csv: model_name = f'{args.base}_{args.model}_f1_{args.bhk}'
    else: model_name = f'{args.base}_{args.model}_f1'
    backbone = f'adults_{args.model}'

    train_data = DisgraphiaDL(args.base, 'train', DEVICE, args.csv, args.bhk)
    validation_data = DisgraphiaDL(args.base, 'validation', DEVICE, args.csv, args.bhk)

    # LOAD MODEL
    if args.model == 'vit':
        wrapper = ViTWrapper(model_name, DEVICE, pen_features=train_data.pen_features)
    elif args.model == 'resnet':
        wrapper = ResnetWrapper(model_name, DEVICE, 2, False, train_data.pen_features)
    else:
        raise Exception(f'{model} is not a model: selecte either vit or resnet.')
    if args.base == 'children': wrapper.load_state(s = f'{backbone}_model_best.pth')
    if args.csv: wrapper.set_csv_model(args.model)
    model = wrapper.get_model()

    # TRAIN SETTINGS
    epochs = 10000
    start_epoch = 0
    batch_size = 32
    best_val_loss = inf
    best_val_f1 = 0.0
    exit_counter = 0
    epsilon = 0.0001 # counter guard precision improovement
    lr = 0.0001

    if args.weighted_loss: loss = CrossEntropyLoss(weight=train_data.get_binary_weights())
    else: loss = CrossEntropyLoss()
    print(train_data.get_binary_weights())
    opt = optim.AdamW(model.parameters(), lr=lr)

    if args.resume:
        start_epoch, best_val_loss, exit_counter, opt_chk, best_val_f1 = wrapper.resume(f'{model_name}_checkpoint.pth')
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
        train_f1 = 0.0
        running_f1 = 0.0

        start = time.time()
        for i in range(0, int(len(train_data) / batch_size) + 1):
            images, classes, pfeat = next(loader)

            opt.zero_grad()
            if not args.csv: preds  = model(images)
            else: preds = model(images, pfeat)

            out = loss(preds, classes)
            out.backward()
            opt.step()

            train_loss += out.item()
            running_loss += out.item()

            preds = np.argmax(preds.cpu().detach().numpy(), axis=1)
            classes = np.asarray(classes.cpu())
            _, _, f1, _ = precision_recall_fscore_support(classes, preds, average='macro', zero_division=1)
            
            train_f1 += f1
            running_f1 += f1

            if i % 5 == 4:
                print(f'Epoch {e + 1} - Batch {i + 1}: Running Loss {running_loss / 5} / Running F1 {running_f1 / 5}', end='\r')
                running_loss = 0.0
                running_f1 = 0.0
        
        train_loss = train_loss / i
        train_f1 = train_f1 / i
        print(f'Epoch {e + 1}: Train Loss {train_loss} - Train F1 {train_f1} - Time {str(timedelta(seconds=time.time() - start))}')

        model.eval()
        with torch.no_grad():
            loader = iter(DataLoader(validation_data, batch_size=len(validation_data), shuffle=False))
            images, classes, pfeat = next(loader)
            if not args.csv: preds  = model(images)
            else: preds = model(images, pfeat)
            out = loss(preds, classes)
            preds = np.argmax(preds.cpu(), axis=1)
            classes = np.asarray(classes.cpu())
            _, _, f1, _ = precision_recall_fscore_support(classes, preds, average='macro', zero_division=1)
            
            print(f"Epoch {e + 1}: Validation Loss {out.item()} - Validation F1 {f1}")
            if f1 > best_val_f1 + epsilon:
                print(f"    !- Validation improovement! {best_val_f1} -> {f1}")
                exit_counter = 0
                best_val_f1 = f1
                is_best = True
            else:
                print(f"    !- No improovement!")
                is_best = False
                exit_counter += 1
            
            wandb.log({
                "train-loss": train_loss,
                "val-loss": out.item(),
                "train-f1": train_f1,
                "val-f1": f1
                })
            # Optional
            wandb.watch(model)
            
            state = {
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'exit_counter': exit_counter,
                'optimizer': opt.state_dict(),
                'best_val_f1': best_val_f1
            }
            wrapper.save_state(state, is_best)
        
        if exit_counter == 50:
            print("Exit")
            break

def test(args, explain):

    # LOAD DATA
    if args.csv: model_name = f'{args.base}_{args.model}_f1_{args.bhk}'
    else: model_name = f'{args.base}_{args.model}_f1'
    test_data = DisgraphiaDL(args.base, 'test', DEVICE, args.csv, args.bhk)

    # LOAD MODEL
    if args.model == 'vit':
        wrapper = ViTWrapper(model_name, DEVICE, pen_features=test_data.pen_features)
    elif args.model == 'resnet':
        wrapper = ResnetWrapper(model_name, DEVICE, 2, False, test_data.pen_features)
    else:
        raise Exception(f'{model} is not a model: selecte either vit or resnet.')
    if args.csv: wrapper.set_csv_model(args.model)
    wrapper.load_state(f'{model_name}_model_best.pth')
    model = wrapper.get_model()

    model.eval()
    with torch.no_grad():
        loader = iter(DataLoader(test_data, batch_size=len(test_data), shuffle=False))
        images, classes, pfeat = next(loader)
        def predict(imgs, pfeat):
            if isinstance(imgs, np.ndarray): imgs = torch.tensor(imgs).to(DEVICE)
            if not args.csv: preds  = model(images)
            else: preds = model(images, pfeat)
            return preds
        out = predict(images, pfeat)
        preds = np.argmax(out.cpu(), axis=1)
        classes = np.asarray(classes.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(classes, preds, average='macro')

        if explain:
            topk = 2
            batch_size = 50
            n_evals = 10000
            masker_blur = shap.maskers.Image("blur(128,128)", images[0].shape)
            class_names = {0: 'not-disgraphia', 1: 'disgraphia'}
            def class_to_names(cls):
                names = []
                for c in cls:
                    names.append(class_names[c])
                return names
            explainer = shap.Explainer(predict, masker_blur, output_names=['not-disgraphia', 'disgraphia'])
            shap_values = explainer(images[:2], max_evals=n_evals, batch_size=batch_size,
                            outputs=shap.Explanation.argsort.flip[:topk])
            shap_values.data = shap_values.data.cpu().numpy()[0]
            shap_values.values = [val for val in np.moveaxis(shap_values.values,-1, 0)]
            shap.image_plot(shap_values=shap_values.values, pixel_values=shap_values.data, 
                            labels=shap_values.output_names, true_labels=class_to_names(classes))
    
    print("Test Results")
    print("---")
    print("Predictions: ", np.asarray(preds.cpu()))
    print("Classes: ", classes)
    print("---")
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
