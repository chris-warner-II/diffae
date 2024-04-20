
# # (0). Imports
from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

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
    dir_figs = 'store/output/diffae/manipulate/'
    os.makedirs(dir_figs,exist_ok=True)

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f'Using device: {device}')

    if device=='cuda':
        os.system('nvidia_smi')


    # # (2). Setup and load in models
    conf = ffhq256_autoenc()
    # print(conf.name)
    model = LitModel(conf)
    state = torch.load(f'{dir_pre}checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device);

    # #  (2b). Conditioning model

    cls_conf = ffhq256_autoenc_cls()
    cls_conf.pretrain.path = dir_pre + cls_conf.pretrain.path           # update path to model weights
    cls_conf.latent_infer_path = dir_pre + cls_conf.latent_infer_path   # update path to model weights

    cls_model = ClsModel(cls_conf)
    state = torch.load(f'{dir_pre}checkpoints/{cls_conf.name}/last.ckpt',
                        map_location='cpu')
    print('latent step:', state['global_step'])
    cls_model.load_state_dict(state['state_dict'], strict=False);
    cls_model.to(device);


    # # (3). Set up data

    # data = conf.make_dataset()
    # batch = data[10]['img'][None]

    data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
    batch = data[0]['img'][None]


    # # (4). Encode
    #Te = 2 # was 250, made smaller to speedup on cpu
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=args.Te)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ori = (batch + 1) / 2
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(xT[0].permute(1, 2, 0).cpu())


    # # (5). Conditioning
    #
    # Can I condition on multiple features? - not yet. Works fine for one.
    # Have to think carefully about what cond2 vectors look like and how
    # to combine them/



    print(CelebAttrDataset.id_to_cls)
    cls_list = ['Big_Nose'] #'Eyeglasses','Wavy_Hair'] 

    cond2 = cls_model.normalize(cond)

    import IPython ; IPython.embed()

    for cls_str in cls_list:

        cls_id = CelebAttrDataset.cls_to_id[cls_str]

        cond2 = cond2 + 0.3 * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
        cond2 = cls_model.denormalize(cond2)

    # # (6). Generate based on conditioning

    # torch.manual_seed(1)
    #Tr = 2 # was 100, made smaller to speed up on cpu
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    img = model.render(xT, cond2, T=args.Tr)
    ori = (batch + 1) / 2
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(img[0].permute(1, 2, 0).cpu())
    plt.savefig(f'{dir_figs}/compare_{"-".join(cls_list)}_Te{args.Te}_Tr{args.Tr}.png')


    # # (7). Plot and save figures
    from torchvision.utils import save_image #*
    save_image(img[0], f'{dir_figs}/output_{"-".join(cls_list)}_Te{args.Te}_Tr{args.Tr}.png')


    # import IPython ; IPython.embed()


if __name__ == "__main__":
    main()


