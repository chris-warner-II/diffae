
# # (0). Imports
from templates import *
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--Te", type=int, default=250, help="Encoder Time Steps")
    parser.add_argument("--Tr", type=int, default=20, help="Render Time Steps")
    args = parser.parse_args()

    return args


def main():

    args = parse_args()


	# # (1). Directory and device
	dir_pre = 'store/models/diffae/'
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
	#Te=2 # was 250 originally, made smaller to run faster on cpu
	xT = model.encode_stochastic(batch.to(device), cond, T=args.Te)
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	ori = (batch + 1) / 2
	ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
	ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
	plt.title(f'Encoding Te={args.Te}')
	plt.savefig(f'{dir_figs}encoding_Te{args.Te}.png')


	# # (5). Decode
	#Td=2 # was 20 originally, made smaller to run faster on cpu
	pred = model.render(xT, cond, T=args.Tr)
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	ori = (batch + 1) / 2
	ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
	ax[1].imshow(pred[0].permute(1, 2, 0).cpu())
	plt.title(f'Decoding Te={args.Te}, Tr={args.Tr}')
	plt.savefig(f'{dir_figs}decoding_Te{args.Te}_Tr{args.Tr}.png')


if __name__ == "__main__":
    main()
