# # (0). Imports
from templates import *
from templates_latent import *
import matplotlib.pyplot as plt


# # (1). Directory and device
dir_pre = 'store/models/diffae/'
dir_figs = 'store/output/diffae/sample/'
os.makedirs(dir_figs,exist_ok=True)

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using device: {device}')


# # (2). Setup and load in model
conf = ffhq256_autoenc_latent()
conf.T_eval = 100
conf.latent_T_eval = 100
print(conf)

# update paths to checkpoints dir with dir_pre
conf.pretrain.path = dir_pre + conf.pretrain.path
conf.latent_infer_path = dir_pre + conf.latent_infer_path
#import IPython ; IPython.embed()

model = LitModel(conf)
state = torch.load(f'{dir_pre}checkpoints/{conf.name}/last.ckpt', map_location='cpu')
print(model.load_state_dict(state['state_dict'], strict=False))
model.to(device);


# # (3). Unconditioned samples from model
torch.manual_seed(4)
Ts = 2 # was 20
Tl = 2 # was 200
imgs = model.sample(8, device=device, T=Ts, T_latent=Tl)


# # (4). Plot and save results
fig, ax = plt.subplots(2, 4, figsize=(4*5, 2*5))
ax = ax.flatten()
for i in range(len(imgs)):
    ax[i].imshow(imgs[i].cpu().permute([1, 2, 0]))
plt.savefig(f'{dir_figs}sample_Ts{Ts}_Tl{Tl}.png')







