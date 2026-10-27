"""Run a trained RIRCI model on one image or a folder of images."""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import src.models as models
from options import Options


def preprocess(file_path, img_size=256):
    img = cv2.imread(file_path)
    assert img is not None, "Cannot read image: {}".format(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...])
    img = F.interpolate(img, size=(img_size, img_size), mode='bilinear',
                        align_corners=False)
    return img


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    found = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in (
                '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
            found.append(full)
    return found


def to_uint8(tensor):
    return (tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)


def main(args):
    args.evaluate = True
    machine = models.__dict__[args.models](datasets=(None, None), args=args)
    machine.model.eval()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    files = collect_images(args.input)
    if not files:
        raise FileNotFoundError("No images found at {}".format(args.input))

    with torch.no_grad():
        for path in files:
            inputs = preprocess(path, args.crop_size).to(machine.device).float()
            outputs = machine.model(machine.norm(inputs))
            imoutput, mask = outputs[0][0], outputs[1][0]

            # Keep pixels outside the predicted watermark region untouched.
            restored = machine.denorm(
                imoutput * mask + machine.norm(inputs) * (1 - mask))

            panel = np.concatenate([
                cv2.cvtColor(to_uint8(inputs), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(to_uint8(restored), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(to_uint8(mask.repeat(1, 3, 1, 1)), cv2.COLOR_RGB2BGR),
            ], axis=1)

            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.output, stem + '.png')
            cv2.imwrite(out_path, panel)
            print("{} -> {}".format(path, out_path))


if __name__ == '__main__':
    parser = Options().init(argparse.ArgumentParser(description='RIRCI inference'))
    parser.add_argument('--input', required=True, type=str,
                        help='image file or a directory of images')
    parser.add_argument('--output', default='inference_output', type=str,
                        help='directory where the [input | result | mask] panels are written')
    main(parser.parse_args())
