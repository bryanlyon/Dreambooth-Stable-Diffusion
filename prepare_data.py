import os, sys
import cv2
from argparse import ArgumentParser
import traceback
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("path",
                    help="folder of images to convert")
parser.add_argument("--export-path",
                    default="$path/db/",
                    help="folder to place face (replaces $path with the input path)")
parser.add_argument("--size",
                    default="1024",
                    type=int,
                    help="square size of the face")
parser.add_argument("--descriptives",
                    default="",
                    help="Descriptive words to add to class word separated by spaces")

opt = parser.parse_args()
opt.export_path = opt.export_path.replace("$path", opt.path)

for cnt, infile in enumerate(tqdm([x for x in sorted(os.listdir(opt.path)) if "jpg" in x.lower() or "png" in x.lower()])):
    if not os.path.exists(opt.export_path):
        os.makedirs(opt.export_path)
    outfile = os.path.join(opt.export_path, f"{str(cnt).zfill(5)}-{opt.descriptives.replace('-','_')}.png")
    infile = os.path.join(opt.path, infile)
    try:
        im = cv2.imread(infile)
        h,w,c = im.shape
        min_side = min(h,w)
        if h > w:
            im = im[:w,:]
        if w > h:
            im = im[:, (w-h)//2:(w-h)//2+h]
        im = cv2.resize(im, (opt.size, opt.size),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(outfile, im)
    except Exception as e:
        print(f"Error reading {infile} due to {e}")
