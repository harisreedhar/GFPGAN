import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--arch', type=str, default='clean')
    parser.add_argument('--channel', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan')
    parser.add_argument('--bg_tile', type=int, default=400)
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--paste_back', action='store_false')
    parser.add_argument('--input_video', type=str, default='inputs/test.mp4')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--save_frames', action='store_true')
    args = parser.parse_args()

    args = parser.parse_args()
    if args.input_video.endswith('/'):
        args.input_video = args.input_video[:-1]
    os.makedirs(args.output_dir, exist_ok=True)

     # background upsampler
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from realesrgan import RealESRGANer
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # set up GFPGAN restorer
    restorer = GFPGANer(
        model_path=args.model_path,
        upscale=args.upscale,
        arch=args.arch,
        channel_multiplier=args.channel,
        bg_upsampler=bg_upsampler)

    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = None
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output_file = os.path.join(args.output_dir, "output.avi")
    output_path = os.path.join(args.output_dir, "frames")
    frame_count = 0

    while True:
        _, frame = cap.read(cv2.IMREAD_COLOR)
        if frame is None:
            break
        else:
            frame_count += 1
            print(f'Processing frame {frame_count} ...')

            cropped_faces, restored_faces, restored_img = restorer.enhance(
                frame,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=args.paste_back)

            # save video
            if video_writer == None:
                frame_size = (restored_img.shape[1], restored_img.shape[0])
                video_writer = cv2.VideoWriter(output_file, codec, fps, frame_size, True)
            video_writer.write(restored_img)

            # save individual frames
            if args.save_frames:
                save_face_name = f'frame_{frame_count}.png'
                save_restore_path = os.path.join(output_path, save_face_name)
                imwrite(restored_img, save_restore_path)

    cap.release()
    cv2.destroyAllWindows()

    print(f'Results are in the [{args.output_dir}] folder.')

if __name__ == '__main__':
    main()
