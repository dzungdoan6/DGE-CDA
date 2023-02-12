import numpy as np
import torch
import pickle
import os
from utils import save_variable


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='plot correlation between domain gap and detection accuracy')
    parser.add_argument('--save-dir', default="", metavar='FILE', help='path to save projections');
    parser.add_argument('--num-projs', default=10, help='number of projections to be generated');
    args = parser.parse_args()
    return args


def rand_projections(dim, num_projections=1000, device="cuda"):
    projections = torch.randn((num_projections, dim)).to(device);
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def main():
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n");

    os.makedirs(args.save_dir, exist_ok=True);
    for dim in [4177920, 8355840, 16711680]:

        projections = rand_projections(dim, num_projections=args.num_projs);
        projections = projections.detach().cpu().numpy();

        save_variable(file_name=args.save_dir + "/projections_n" + str(args.num_projs) + "_d" + str(dim), data=projections);

    print("Done!!!");

if __name__ == "__main__":
    main();