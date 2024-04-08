from templates import *
from templates_latent import *

if __name__ == '__main__':

    verbose = True

    # train the autoenc moodel
    # this requires V100s.
    if verbose: print('Train ffhq128 autoenc 72M model.')
    gpus = [0, 1]#, 2, 3]
    conf = ffhq128_autoenc_72M()
    train(conf, gpus=gpus, verbose=verbose)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    if verbose: print('Infer latents for DPM.')
    gpus = [0, 1]#, 2, 3]
    conf.eval_programs = ['infer']
    train(conf, gpus=gpus, mode='eval', verbose=verbose)

    # train the latent DPM
    # NOTE: only need a single gpu
    if verbose: print('Train latent DPM.')
    gpus = [0]
    conf = ffhq128_autoenc_latent() # Might have to train pretrained model to 72M in here (CW)
    train(conf, gpus=gpus, verbose=verbose)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    if verbose: print('Unconditional sampling score.')
    gpus = [0, 1]#, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    train(conf, gpus=gpus, mode='eval', verbose=verbose)