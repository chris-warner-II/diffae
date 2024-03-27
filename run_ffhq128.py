from templates import *
from templates_latent import *

if __name__ == '__main__':

    verbose = True

    # train the autoenc moodel
    # this requires V100s.
    if verbose: print('Train ffhq128 autoenc model.')
    gpus = [0, 1, 2, 3]
    conf = ffhq128_autoenc_130M()
     if verbose: print(f'conf =  {conf}')
    train(conf, gpus=gpus)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    if verbose: print('Infer latents for DPM.')
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval')

    # train the latent DPM
    # NOTE: only need a single gpu
    if verbose: print('Train latent DPM.')
    gpus = [0]
    conf = ffhq128_autoenc_latent()
    train(conf, gpus=gpus)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    if verbose: print('Unconditional sampling score.')
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval')