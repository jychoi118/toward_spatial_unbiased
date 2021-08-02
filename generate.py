import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import math


def generate(args, g_ema, device, mean_latent):
    radius = 32
    with torch.no_grad():
        g_ema.eval()
        sample_z = torch.randn(args.sample, args.latent, device=device)
        for i in tqdm(range(args.pics)):
            dh = math.sin(2 * math.pi * (i / args.pics)) * radius
            dw = math.cos(2 * math.pi * (i / args.pics)) * radius
            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent, shift_h=dh, shift_w=dw)

            for j in range(args.sample):
                utils.save_image(
                    sample[j].unsqueeze(0),
                    f"generate/{args.name}/{str(j).zfill(3)}_{str(i).zfill(4)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=4,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=100, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="550000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--position', type=str, default='mspe', help='pe options (none | sspe | mspe | rand_mspe)')
    parser.add_argument('--kernel_size', type=int, default=3)

    args = parser.parse_args()


    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, position=args.position, kernel_size=args.kernel_size, scale=1.0, device=device,
    ).to(device)
    checkpoint = torch.load(f'checkpoint/{args.name}/{args.ckpt}')

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
