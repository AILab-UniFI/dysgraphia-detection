import argparse

from training import train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--resume', '-r', action="store_true",
                        help="resume training")
    parser.add_argument('--base', '-b', choices=['children', 'adults'], default='children',
                        help="set which dataset to load, either 'adults' or 'children'")
    parser.add_argument('--model', '-m', choices=['resnet', 'vit'], default='resnet',
                        help="set which model to use / train, either 'resnet' or 'vit'")  
    parser.add_argument('--csv', action="store_true",
                        help="use pen features stored in csv")
    parser.add_argument('--bhk', choices=['binary', 'float', 'double'], default='binary',
                        help="decide which csv to load, either 'binary, 'float' or 'double'")
    parser.add_argument('--labels', '-l', choices=['certified', 'expert', 'professors'], default='certified',
                        help="decide which labels to load, either 'certified, 'expert' or 'professors'")
    parser.add_argument('--split', '-s', default=0,
                        help="which train/val/test split to load")
    parser.add_argument('--weighted_loss', '-wl', action="store_true",
                        help="either using a weighted CrossEntropyLoss or not")
    parser.add_argument('--freeze', '-brr', action="store_true",
                        help="freeze all layers for training but the head")
    parser.add_argument('--explain', '-ex', action="store_true",
                        help="print explainer SHAP results")     
    parser.add_argument('--test', '-t', action="store_true",
                        help="only testing mode")                

    args = parser.parse_args()
    print(args)

    if not args.test: 
        train(args)
    test(args, explain=args.explain)

