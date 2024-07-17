import argparse

namespase = {
    'seed': 1234,
    'n_epoch': 200,
    'n_batch': 2,
    'lr': 0.001,
    'save_path': '../data/model/p_sentence_classification.pth',
}


parser = argparse.ArgumentParser(description='arg test')
parser.add_argument('--seed', required=True)

args = parser.parse_args()

argparse.Namespace(**namespase)

print(args)
