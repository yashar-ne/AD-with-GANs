import os
from typing import List, Optional

import PIL.Image
import torch

import dnnlib
import legacy

device = torch.device('cuda')


def generate_stylegan2_image(
        network_pkl,
        outdir,
        class_idx: Optional[int],
        truncation_psi: float,
        noise_mode: str,
        seeds: Optional[List[int]]
):
    print(f"Generating Images, output in {outdir}")
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        print(G.num_ws, G.w_dim)

    os.makedirs(outdir, exist_ok=True)
    projected_w = None
    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        # ws = np.load(projected_w)['w']
        # ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        ws = torch.randn(1, 14, 512, device=device)
        print(ws.shape[1:])
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').show()
            # img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        print('error: --seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            print('error: Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        z = torch.randn(14, 512, device=device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').show()


generate_stylegan2_image(
    network_pkl='/home/yashar/git/AD-with-GANs/data/stylegan2-celebahq-256x256.pkl',
    outdir='/home/yashar/git/AD-with-GANs/data/test_out',
    class_idx=None,
    truncation_psi=0.7,
    noise_mode='const',
    seeds=[149209]
)
