import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer

import datetime

def ESRGAN(model, image_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument('--block', type=int, default=23, help='num_block in RRDB')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    if 'RealESRGAN_x4plus_anime_6B.pth' in args.model_path:
        args.block = 6
    elif 'RealESRGAN_x2plus.pth' in args.model_path:
        args.netscale = 2

    '''
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.block, num_grow_ch=32, scale=args.netscale)

    upsampler = RealESRGANer(
        scale=args.netscale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)
    
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    '''
    os.makedirs(args.output, exist_ok=True)


    upsampler = model

    #path = image_path

    #for idx, path in enumerate(paths):
    imgname, extension = os.path.splitext(os.path.basename(image_path))
    print(datetime.datetime.now())
    print('Testing', imgname)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

        h, w = img.shape[0:2]
    if max(h, w) > 1000 and args.netscale == 4:
        import warnings
        warnings.warn('The input image is large, try X2 model for better performace.')
    if max(h, w) < 500 and args.netscale == 2:
        import warnings
        warnings.warn('The input image is small, try X4 model for better performace.')

    try:
        output, _ = upsampler.enhance(img, outscale=args.outscale)
    except Exception as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        if args.ext == 'auto':
            extension = extension[1:]
        else:
            extension = args.ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output)
    print(datetime.datetime.now())
    return save_path