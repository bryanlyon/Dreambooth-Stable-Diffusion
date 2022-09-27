import os, sys
import cv2
from argparse import ArgumentParser
import traceback
from tqdm import tqdm

minsize = 512
parser = ArgumentParser()
parser.add_argument("path",
                    help="folder of images to convert")
parser.add_argument("--export-path",
                    default="$path/db/",
                    help="folder to place face (replaces $path with the input path)")
parser.add_argument("--descriptives",
                    default="",
                    help="Descriptive words to add to class word separated by spaces")

opt = parser.parse_args()
opt.export_path = opt.export_path.replace("$path", opt.path)

for cnt, infile in enumerate(tqdm([x for x in sorted(os.listdir(opt.path)) if "jpg" in x.lower()])):
    if not os.path.exists(opt.export_path):
        os.makedirs(opt.export_path)
    outfile = os.path.join(opt.export_path, f"{str(cnt).zfill(5)}-{opt.descriptives}.png")
    infile = os.path.join(opt.path, infile)
    try:
        im = cv2.imread(infile)
        h,w,c = im.shape
        min_side = min(h,w)
        resize = minsize / min_side
        new_h = int(h*resize)
        new_w = int(w*resize)
        im = cv2.resize(im, (int(w*resize), int(h*resize)))
        if new_h > 512:
            im = im[:512,:]
        if new_w > 512:
            im = im[:, (new_w-512)//2:(new_w-512)//2+512]
        cv2.imwrite(outfile, im)
    except Exception as e:
        print(f"Error reading {infile} due to {e}")
