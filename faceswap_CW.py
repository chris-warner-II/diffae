
# # (0). Imports
from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--Te", type=int, default=250, help="Encoder Time Steps")
    parser.add_argument("--Tr", type=int, default=100, help="Render Time Steps")
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # # (1). Directory and device
    dir_pre = 'store/models/diffae/'
    dir_figs = 'store/output/diffae/faceswap/'
    os.makedirs(dir_figs, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(f'Using device: {device}')

    if device=='cuda':
        os.system('nvidia_smi')


    # # (2). Setup and load in models
    conf = celeba64d2c_autoenc()
    conf.name = 'celeba64d2c_autoenc_embeddings_zsem_train'
    conf.data_name = 'celebaembeddingstestlmdb'

    # print(conf.name)
    model = LitModel(conf)
    state = torch.load(f'{dir_pre}checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)


    # # (3). Set up data (source and target)
    data = conf.make_dataset()

    #idx = []
    for _ in tqdm(range(100)):
        t,s = random.sample(range(len(data)), 2) # randomly sample source & target

        target_image = data[t]['img'][None].to(device)
        target_embedding = data[t]['embed'].to(device)
        target_fname = data[t]['fname']
        #
        source_image = data[s]['img'][None].to(device)
        source_embedding = data[s]['embed'].to(device)
        source_fname = data[s]['fname']

        # # (4). Encode
        cond = model.encode(target_image) + target_embedding
        xT = model.encode_stochastic(target_image, cond, T=args.Te)

        # # (5). Conditioning on another identity in test set - FaceSwap
        cond2 = cond - target_embedding + source_embedding
        swap_img = model.render(xT, cond2, T=args.Tr)

        # # (6). Plot and save figures
        plt.figure( figsize=(10,5) )
        plt.subplot(1,3,1)
        plt.imshow(source_image[0].permute(1, 2, 0).cpu())
        plt.title(f"Source: {source_fname}")
        plt.xticks([])
        plt.yticks([])
        #
        plt.subplot(1, 3, 2)
        plt.imshow(swap_img[0].permute(1, 2, 0).cpu())
        plt.title(f"Swap")
        plt.xticks([])
        plt.yticks([])
        #
        plt.subplot(1, 3, 3)
        plt.imshow(target_image[0].permute(1, 2, 0).cpu())
        plt.title(f"Target: {target_fname}")
        plt.xticks([])
        plt.yticks([])
        #
        plt.suptitle(f"Encode T {args.Te} : Render T {args.Tr}")
        plt.savefig(f'{dir_figs}/swap_src{source_fname[:-4]}_tgt{target_fname[:-4]}_Te{args.Te}_Tr{args.Tr}.png')

    #import IPython; IPython.embed()


if __name__ == "__main__":
    main()


