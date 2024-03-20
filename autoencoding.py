
# # (0). Imports
from templates import *
import matplotlib.pyplot as plt
import os


# # (1). Directory and device
dir_pre = 'store/models/'
dir_figs = 'store/output/diffae/autoencoding/'
os.makedirs(dir_figs,exist_ok=True)

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using device: {device}')



# # (2). Setup and load in model
conf = ffhq256_autoenc()
print(conf)
model = LitModel(conf)
print(model)
state = torch.load(f'{dir_pre}checkpoints/{conf.name}/last.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

# # (3). Set up data
data = ImageDataset('imgs_align', image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
batch = data[0]['img'][None]

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
plt.title(f'Encoding Te={Te}')
plt.savefig(f'{dir_figs}encoding_Te{Te}.png')


# # (5). Decode
Td=2 # was 20 originally, made smaller to run faster on cpu
pred = model.render(xT, cond, T=Td)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ori = (batch + 1) / 2
ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
ax[1].imshow(pred[0].permute(1, 2, 0).cpu())
plt.title(f'Decoding Te={Te}, Td={Td}')
plt.savefig(f'{dir_figs}decoding_Te{Te}_Td{Td}.png')


