from templates import *
from templates_latent import *

if __name__ == '__main__':

    verbose = True

    # train the autoenc model
    # this can be run on 2080Ti's.
    gpus = [0, 1, 2, 3]
    conf = celeba64d2c_autoenc()
    #
    conf.batch_size = 32
    conf.data_name = 'celebaembeddingslmdb'
    conf.name = 'celeba64d2c_autoenc_embeddings_norm_resume'
    #conf.fid_cache = conf.fid_cache + 'b' # to keep two jobs from crashing into eachother.
    #       not working. Add it directly into eval_fid function in metrics.py
    #
    #import IPython; IPython.embed()
    print('First time thru train in run_celeba64')
    train(conf, gpus=gpus, verbose=verbose)

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['infer']
    print('2nd time thru train in run_celeba64')
    train(conf, gpus=gpus, mode='eval', verbose=verbose)

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = celeba64d2c_autoenc_latent()
    print('3rd time thru train in run_celeba64')
    train(conf, gpus=gpus, verbose=verbose)

    # unconditional sampling score
    # NOTE: a lot of gpus can speed up this process
    gpus = [0, 1, 2, 3]
    conf.eval_programs = ['fid(10,10)']
    print('4th time thru train in run_celeba64')
    train(conf, gpus=gpus, mode='eval', verbose=verbose)