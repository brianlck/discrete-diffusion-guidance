from cleanfid import fid
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path to the directory containing the images')

args = parser.parse_args()
path = args.path

score = fid.compute_fid(
    path,
    dataset_name='cifar10',
    dataset_res=32,
    dataset_split='train'
)

print("Score", score)
