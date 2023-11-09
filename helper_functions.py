# Credit to Pu Li https://github.com/Paul-LiPu/DeepWhistle
import os
import fnmatch


# find file  with certain pattern.
def findfiles(path, fnmatchex='*.*'):
    result = []
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnmatchex):
            fullname = os.path.join(root, filename)
            result.append(fullname)
    return result

# Get all subdirs in one directory
def list_all_dir(path):
    result = []
    files = os.listdir(path)
    for file in files:
        m = os.path.join(path, file)
        if os.path.isdir(m):
            result.append(m)
    return result

# Get .wav files in one directory
def find_wav_files(path):
    return findfiles(path, fnmatchex='*.wav')


# check if dir exists. if not, create it.
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Substitue postfix .bin to .wav.
def bin2wav_filename(bin_file):
    bin_filename = os.path.basename(bin_file)
    bin_name, ext = os.path.splitext(bin_filename)
    return bin_name + '.wav'
