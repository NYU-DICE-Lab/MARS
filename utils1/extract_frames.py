'''
For HMDB51 and UCF101 datasets:

Code extracts frames from video at a rate of 25fps and scaling the
larger dimension of the frame is scaled to 256 pixels.
After extraction of all frames write a "done" file to signify proper completion
of frame extraction.

Usage:
  python extract_frames.py video_dir frame_dir
  
  video_dir => path of video files
  frame_dir => path of extracted jpg frames

'''

import sys, os, pdb
import numpy as np
import subprocess
from tqdm import tqdm
import os
import cv2
import h5py
import multiprocessing
import shutil

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def extract_hdf5(path):
  f_list = os.listdir(os.path.join(path))
  hdf5_path = os.path.join("/".join(path.split("/")[:-1]), path.split("/")[-1]+".hdf5")
  print("HDF5 path =", hdf5_path)
  with h5py.File(hdf5_path, "w") as f: #initialize hdf5
    for f1 in f_list:
      dset = f.create_dataset(f1, (len(f_list), 256, 341, 3), "f")
      for i, f2 in enumerate(f_list):
          file_path = os.path.join(path, f1)
          img = cv2.imread(file_path)
          dset[i] = img[:256,:341,:]
    f.close()
       
def extract(vid_dir, frame_dir, start, end, redo=False):
  class_list = sorted(os.listdir(vid_dir))[start:end]

  print("Classes =", class_list)
  
  for ic,cls in enumerate(class_list): 
    vlist = sorted(os.listdir(os.path.join(vid_dir,cls)))
    print("")
    print(ic+1, len(class_list), cls, len(vlist))
    print("")
    for v in tqdm(vlist):
      outdir = os.path.join(frame_dir, cls, v[:-4])
      
      # Checking if frames already extracted
      if os.path.isfile( os.path.join(outdir, 'done') ) and not redo: continue
      try:  
        os.system('mkdir -p "%s"'%(outdir))
        # check if horizontal or vertical scaling factor
        o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
        lines = o.splitlines()
        width = int(lines[0].split('=')[1])
        height = int(lines[1].split('=')[1])
        resize_str = '-1:256' if width>height else '256:-1'

        # extract frames
        os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%( os.path.join(vid_dir, cls, v), resize_str, os.path.join(outdir, '%05d.jpg')))
        nframes = len([ fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname)==9])
        if nframes==0: raise Exception 
        extract_hdf5(outdir)
        shutil.rmtree(outdir)
        os.system('touch "%s"'%(os.path.join(outdir, 'done') ))
      except Exception as e:
        print("ERROR", cls, v, e)

if __name__ == '__main__':
  vid_dir   = sys.argv[1]
  frame_dir = sys.argv[2]
  start     = int(sys.argv[3])
  end       = int(sys.argv[4])
  thread_num = 10

  argument_list = [(vid_dir, frame_dir, i[0], i[-1], True) for i in list(split(list(range(102)), thread_num))]
  p = multiprocessing.Pool(thread_num)
  p.starmap(extract, argument_list)
  p.close()
  p.join()
  for i in argument_list:
    print(i)