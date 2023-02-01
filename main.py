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
    parser.add_argument('--weighted_loss', '-wl', action="store_true",
                        help="either using a weighted CrossEntropyLoss or not")
    parser.add_argument('--explain', '-ex', action="store_true",
                        help="print explainer SHAP results")                     

    args = parser.parse_args()
    print(args)

    train(args)
    test(args, explain=args.explain)

