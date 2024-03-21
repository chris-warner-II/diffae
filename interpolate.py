
# # (0). Imports
from templates import *
import matplotlib.pyplot as plt
import numpy as np


# # (1). Directory and device
dir_pre = 'store/models/diffae/'
dir_figs = 'store/output/diffae/interpolate/'
os.makedirs(dir_figs,exist_ok=True)

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using device: {device}')


# # (2). Setup and load in model
conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'{dir_pre}checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);


# # (3). Set up data
data = ImageDataset('imgs_interpolate', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = torch.stack([
    data[0]['img'],
    data[1]['img'],
])

# plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)
# plt.show()
# import IPython ; IPython.embed()



# # (4). Encode
cond = model.encode(batch.to(device))
Te=2 # was 250 originally, made smaller to run faster on cpu
xT = model.encode_stochastic(batch.to(device), cond, T=Te)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
plt.savefig(f'{dir_figs}encoding_Te{Te}.png')


# # (5). Interpolate
# 
# Semantic codes are interpolated using convex combination, while stochastic 
# codes are interpolated using spherical linear interpolation.

alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device)
intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

theta = torch.arccos(cos(xT[0], xT[1]))
x_shape = xT[0].shape
intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + 
          torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
intp_x = intp_x.view(-1, *x_shape)

Ti = 2 # was 20, made smaller for speedup on cpu
pred = model.render(intp_x, intp, T=Ti)


# # (6). Plot interpolation results

# torch.manual_seed(1)
fig, ax = plt.subplots(1, 10, figsize=(5*10, 5))
for i in range(len(alpha)):
    ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
plt.savefig(f'{dir_figs}interpolate_Te{Te}_Ti{Ti}.png')





