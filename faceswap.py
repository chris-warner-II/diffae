
# # (0). Imports
from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

from torch.nn.functional import normalize
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


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = normalize(a, dim=0)
    b = normalize(b, dim=0)
    return (a * b).sum()


def main():

    args = parse_args()

    # # (1). Directory and device
    dir_pre = 'store/models/diffae/'
    #dir_figs = 'store/output/diffae/faceswap/embed_norm_model'
    dir_figs = 'store/output/diffae/faceswap/embed_norm_zsem_norm_model'
    os.makedirs(dir_figs, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(f'Using device: {device}')

    if device=='cuda':
        os.system('nvidia-smi')


    # # (2). Setup and load in models
    conf = celeba64d2c_autoenc()
    #conf.name = 'celeba64d2c_autoenc_embeddings_norm'
    conf.name = 'celeba64d2c_autoenc_embeddings_norm_resume12M_zsem_norm'
    conf.data_name = 'celebaembeddingslmdb'

    # print(conf.name)
    model = LitModel(conf)
    state = torch.load(f'{dir_pre}checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)


    # # (3). Set up data (source and target)
    data = conf.make_dataset()

    for _ in tqdm(range(100)):
        t,s = random.sample(range(len(data)), 2) # randomly sample source & target

        target_image = data[t]['img'][None].to(device)
        target_embedding = data[t]['embed'].unsqueeze(0).to(device)
        #
        source_image = data[s]['img'][None].to(device)
        source_embedding = data[s]['embed'].unsqueeze(0).to(device)
        #
        try:
            target_fname = data[t]['fname'] # to label plots and files by image file name
            source_fname = data[s]['fname']
        except:
            target_fname = str(t).zfill(7) + '.idx' # in case 'fname' not in lmdb dataset
            source_fname = str(s).zfill(7) + '.idx'


        # PLOTTING.
        #
        # Put values from 0..1 to be between -1 & 1 for plotting
        src_img = (source_image + 1) / 2
        tgt_img = (target_image + 1) / 2

        # # (6). Plot and save figures
        weights = [0, .25, .5, .75, 1] # interpolation weights between src & tgt
        C=len(weights)+1
        plt.figure( figsize=(10,5) )
        plt.subplot(3,C+1,C+1)
        plt.imshow(src_img[0].permute(1, 2, 0).cpu())
        plt.title(f"GT Img: {source_fname[:-4]}")
        plt.xticks([])
        plt.yticks([])
        #plt.ylabel('xT <= src')
        #
        plt.subplot(3,C+1,2*(C+1)+1)
        plt.imshow(tgt_img[0].permute(1, 2, 0).cpu())
        plt.title(f"GT Img: {target_fname[:-4]}")
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('xT <= tgt')
        #
        #
        plt.subplot(3,C+1,3*(C+1))
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        #
        plt.subplot(3,C+1,1)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('xT <= src')
        #
        plt.subplot(3,C+1,2*(C+1))
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        #
        plt.subplot(3,C+1,(C+1)+1)
        plt.axis('square')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('xT <= 50/50')



        theta_emb = torch.arccos(cos(target_embedding, source_embedding))

        xT_src = model.encode_stochastic(source_image, source_embedding, T=args.Te)
        xT_tgt = model.encode_stochastic(target_image, target_embedding, T=args.Te)
        theta_xT = torch.arccos(cos(xT_src, xT_tgt))

        # # Interpolate between xT_tgt and xT_src with a 50/50 mix - using spherical linear interp.
        x_shape = xT_src.shape
        xT_5050 = (torch.sin(0.5 * theta_xT) * xT_src.flatten(0, 2)[None] + 
                   torch.sin(0.5 * theta_xT) * xT_tgt.flatten(0, 2)[None]) / torch.sin(theta_xT)
        xT_5050 = xT_5050.view(*x_shape)


        

        dot_e = np.dot(source_embedding.cpu(),target_embedding.cpu().T).item()
        dot_x = np.dot(normalize(xT_src.cpu().flatten(),p=2,dim=0),
                       normalize(xT_tgt.cpu().flatten(),p=2,dim=0).T).item()

        #dot_5 = np.dot(normalize(xT_src.flatten(),p=2,dim=0),
        #               normalize(xT_5050.flatten(),p=2,dim=0).T)


        #import IPython; IPython.embed()

        # # (4). Encode
        #
        # (a). holding stochastic embeddings (xT) fixed at target or source and varying 
        #      semantic embeddings (normalized insightface) using spherical linear interp.
        for i,alpha in enumerate(weights):
            #cond = normalize((1-alpha)*target_embedding + alpha*source_embedding, p=2, dim=1) # linear interpolation normed to unit sphere
            #cond = (1-alpha)*target_embedding + alpha*source_embedding

            # Normed insightface embeddings are interpolated using spherical linear interpolation.
            cond = (torch.sin((1 - alpha) * theta_emb) * target_embedding + 
                      torch.sin(alpha * theta_emb) * source_embedding) / torch.sin(theta_emb)

            xT_src = model.encode_stochastic(source_image, cond, T=args.Te)
            xT_tgt = model.encode_stochastic(target_image, cond, T=args.Te)


            swap_src = model.render(xT_src, cond, T=args.Tr)
            swap_tgt = model.render(xT_tgt, cond, T=args.Tr)
            swap_50 = model.render(xT_5050, cond, T=args.Tr)

            #print(f'{i}, shape swaps = {swap_src.shape}, {swap_tgt.shape}')
            #print(f'  Len cond = {np.sqrt( (cond**2).sum() )}')
            

            # PLOTTING CONTINUED.
            #
            #
            plt.subplot(3,C+1,2+i)
            plt.imshow(swap_src[0].permute(1, 2, 0).cpu())
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f"{alpha:.2f}src")
            #
            plt.subplot(3,C+1,(C+1)+2+i)
            plt.imshow(swap_50[0].permute(1, 2, 0).cpu())
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f"{alpha:.2f}src")
            #
            plt.subplot(3,C+1,2*(C+1)+2+i)
            plt.imshow(swap_tgt[0].permute(1, 2, 0).cpu())
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f"{1-alpha:.2f}tgt")





        # cond = target_embedding # + model.encode(target_image)
        # xT = model.encode_stochastic(target_image, cond, T=args.Te)

        # # # (5). Conditioning on another identity in test set - FaceSwap
        # cond2 = 0.5*source_embedding + 0.5*target_embedding #cond #- target_embedding + source_embedding
        # swap_img = model.render(xT_tgt, cond2, T=args.Tr)

        # cond_src = source_embedding #+ model.encode(source_image)
        # xT_src = model.encode_stochastic(source_image, cond_src, T=args.Te)
        # gen_src_img = model.render(xT_src, cond_src, T=args.Tr)

        # cond_tgt = target_embedding #+ model.encode(target_image)
        # xT_tgt = model.encode_stochastic(target_image, cond_tgt, T=args.Te)
        # gen_tgt_img = model.render(xT_tgt, cond_tgt, T=args.Tr)



        #
        plt.suptitle(f"Encode T {args.Te} : Render T {args.Tr} : $dot_e$={dot_e:.3f} : $dot_x$={dot_x:.3f}")
        plt.savefig(f'{dir_figs}/swap_src{source_fname[:-4]}_tgt{target_fname[:-4]}_Te{args.Te}_Tr{args.Tr}.png')

    #import IPython; IPython.embed()


if __name__ == "__main__":
    main()


