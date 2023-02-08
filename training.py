import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, mean_squared_error
import shap
import torch
from torch.utils.data import DataLoader
import random
from torch.nn import CrossEntropyLoss, Sigmoid, BCELoss, MSELoss, ReLU
import torch.optim as optim
import time
from math import inf
import wandb
from datetime import timedelta
from mlxtend.evaluate import accuracy_score

wandb.init(project="dyslexia", entity="ailab-unifi")
random.seed(42)

from model import ViTWrapper, ResnetWrapper
from data import DysgraphiaDL
from path import *

def train(args):

    # LOAD DATA
    if args.csv: model_name = f'{args.base}_{args.model}_{args.bhk}_{args.labels}_split{args.split}'
    else: model_name = f'{args.base}_{args.model}_{args.labels}_split{args.split}'
    backbone = f'adults_{args.model}'
    out_classes = 2
    if args.labels == 'professors': out_classes = 1

    train_data = DysgraphiaDL(args.base, 'train', DEVICE, args.csv, args.bhk, args.labels, args.split)
    validation_data = DysgraphiaDL(args.base, 'validation', DEVICE, args.csv, args.bhk, args.labels, args.split)
    # print(train_data.pen_features)
    # LOAD MODEL
    if args.model == 'vit':
        wrapper = ViTWrapper(model_name, DEVICE, 2, False, pen_features=train_data.pen_features)
    elif args.model == 'resnet':
        wrapper = ResnetWrapper(model_name, DEVICE, 2, False, pen_features=train_data.pen_features)
    else:
        raise Exception(f'{model} is not a model: selecte either vit or resnet.')
    if args.base == 'children': wrapper.load_state(s = f'{backbone}_model_best.pth')
    if out_classes == 1: wrapper.binary()
    if args.csv: wrapper.set_csv_model(args.model)
    if args.freeze: wrapper.freeze()
    model = wrapper.get_model()

    print(model)

    # TRAIN SETTINGS
    epochs = 1000
    start_epoch = 0
    batch_size = 8
    best_val_loss = inf
    best_val_crit = inf
    exit_counter = 0
    epsilon = 0.001 # counter guard precision improovement
    lr = 0.00001

    if out_classes != 1:
        if args.weighted_loss: loss = CrossEntropyLoss(weight=train_data.get_binary_weights())
        else: loss = CrossEntropyLoss()
    else:
        act = Sigmoid()
        func = BCELoss()
        loss = lambda x, y: func(torch.reshape(act(x), (-1,)), y.type(torch.float32))

    opt = optim.AdamW(model.parameters(), lr=lr)

    if args.resume:
        start_epoch, best_val_loss, exit_counter, opt_chk, best_val_crit = wrapper.resume(f'{model_name}_checkpoint.pth')
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
        train_crit = 0.0
        running_crit = 0.0

        start = time.time()
        for i in range(0, int(len(train_data) / batch_size) + 1):
            images, classes, pfeat = next(loader)

            opt.zero_grad()
            if not args.csv: preds = model(images)
            else: preds = model(images, pfeat)

            out = loss(preds, classes)
            out.backward()
            opt.step()

            train_loss += out.item()
            running_loss += out.item()

            classes = np.asarray(classes.cpu())

            if out_classes != 1:
                preds = np.argmax(preds.cpu().detach().numpy(), axis=1)
                _, _, crit, _ = precision_recall_fscore_support(classes, preds, average=None, zero_division=1)
            else:
                preds = torch.reshape(act(preds), (-1,)).cpu().detach().numpy()
                crit = mean_squared_error(classes, preds, squared=False)
            train_crit += crit
            running_crit += crit

            if i % 5 == 4:
                print(f'Epoch {e + 1} - Batch {i + 1}: Running Loss {running_loss / 5} / Running Criteria {running_crit / 5}', end='\r')
                running_loss = 0.0
                running_crit = 0.0
        
        train_loss = train_loss / i
        train_crit = train_crit / i
        print(f'Epoch {e + 1}: Train Loss {train_loss} - Train Criteria {train_crit} - Time {str(timedelta(seconds=time.time() - start))}')

        model.eval()
        with torch.no_grad():
            loader = iter(DataLoader(validation_data, batch_size=len(validation_data), shuffle=False))
            images, classes, pfeat = next(loader)
            if not args.csv: preds  = model(images)
            else: preds = model(images, pfeat)
            out = loss(preds, classes)
            
            classes = np.asarray(classes.cpu())
            
            if out_classes != 1:
                _, _, f1, _ = precision_recall_fscore_support(classes, preds, average=None, zero_division=1)
                crit = f1[1]
            else:
                preds = torch.reshape(act(preds), (-1,)).cpu()
                crit = mean_squared_error(classes, preds, squared=False)

            val_crit = crit
            print(f"Epoch {e + 1}: Validation Loss {out.item()} - Validation Criteria {val_crit}")
            if val_crit < best_val_crit - epsilon:
                print(f"    !- Validation improovement! {best_val_crit} -> {val_crit}")
                exit_counter = 0
                best_val_crit = val_crit
                is_best = True
            else:
                print(f"    !- No improovement!")
                is_best = False
                exit_counter += 1
            
            wandb.log({
                "train-loss": train_loss,
                "val-loss": out.item(),
                "train-criteria": train_crit,
                "val-criteria": val_crit
                })
            # Optional
            wandb.watch(model)
            
            state = {
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'exit_counter': exit_counter,
                'optimizer': opt.state_dict(),
                'best_val_crit': best_val_crit
            }
            wrapper.save_state(state, is_best)
        
        if exit_counter == 20:
            print("Exit")
            break

def test(args, explain):

    # LOAD DATA
    if args.csv: model_name = f'{args.base}_{args.model}_{args.bhk}_{args.labels}_split{args.split}'
    else: model_name = f'{args.base}_{args.model}_{args.labels}_split{args.split}'
    test_data = DysgraphiaDL(args.base, 'test', DEVICE, args.csv, args.bhk, args.labels, args.split)

    out_classes = 2
    if args.labels == 'professors': out_classes = 1

    # LOAD MODEL
    if args.model == 'vit':
        wrapper = ViTWrapper(model_name, DEVICE, out_classes, pen_features=test_data.pen_features)
    elif args.model == 'resnet':
        wrapper = ResnetWrapper(model_name, DEVICE, out_classes, False, test_data.pen_features)
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

    if out_classes != 1: preds = np.argmax(out.cpu(), axis=1)
    else: 
        act = Sigmoid()
        preds = torch.reshape(act(out), (-1,)).cpu()
    classes = np.asarray(classes.cpu())

    print(f"Test Results {model_name}")
    print("---")
    print("Predictions: ", np.asarray(preds.cpu()))
    print("Classes: ", classes)
    print("---")

    if out_classes != 1:
        precision, recall, f1, _ = precision_recall_fscore_support(classes, preds, average='macro')
        accuracy = accuracy_score(classes, preds)

        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("F1:", round(f1, 3))
        print("Accuracy:", round(accuracy, 3))
    
    else:
        mse = mean_squared_error(classes, preds, squared=False)
        print("Mean Squarred Error:", round(mse, 3))

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
    
    
