import h5py



import wavio
import argparse
import numpy as np
import helper_functions as wav2spec
from write_images import write_images
from silbidopy.readBinaries import tonalReader
from silbidopy.render import getSpectrogram, getAnnotationMask



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_hdf5', type=str, help='The hdf5 file to be split into two, one with the positive data and the other with the negative data')
    parser.add_argument('output_dir', type=str, help='The folder into which the two new hdf5 will be written')
    parser.add_argument('--positive_file_name', type=str, default="pos.hdf5", help='The name of the output hdf5 file that contains the positive data')
    parser.add_argument('--negative_file_name', type=str, default="neg.hdf5", help='The name of the output hdf5 file that contains the negative data')
    parser.add_argument('--block_size', type=int, default=128, help='How many examples may be held in memory at one time before a write occurres')
    config = parser.parse_args()    

    half_block_size = config.block_size // 2

    input_file = h5py.File(config.input_hdf5)

    height, width = input_file['data'].shape[1], input_file['data'].shape[2]
    
    # The hdf5 outputs for both positive (True) and negative (False) examples
    hdf5s = {
        True: h5py.File(config.output_dir + config.positive_file_name, 'w'),
        False: h5py.File(config.output_dir + config.negative_file_name, 'w')
        }
    have_not_added_data = {
        True: True,
        False : True
    }
    block_idx = {
        True: 0,
        False: 0
    }
    spectrogram_block = {
        True: np.zeros((half_block_size, height, width), dtype="f4"),
        False: np.zeros((half_block_size, height, width), dtype="f4")
    }
    mask_block = {
        True: np.zeros((half_block_size, height, width), dtype="f4"),
        False: np.zeros((half_block_size, height, width), dtype="f4")
    }

    num_processed = 0
    added = 0
    for spectrogram, mask, flag in zip(input_file['data'], input_file['label'], input_file['positive_flag']):

        flag = True if flag == 1 else False

        spectrogram_block[flag][block_idx[flag]] = spectrogram
        mask_block[flag][block_idx[flag]] = mask

        # Save block of patches to hdf5
        if (block_idx[flag] == half_block_size - 1 or num_processed == input_file['data'].shape[0] - 1) and have_not_added_data[flag]:
            # create datasets
            hdf5s[flag].create_dataset('data', data=spectrogram_block[flag][:block_idx[flag]+1], compression="gzip", chunks=True, maxshape=(None, height, width))
            hdf5s[flag].create_dataset('label', data=mask_block[flag][:block_idx[flag]+1], compression="gzip", chunks=True, maxshape=(None,height, width))
            have_not_added_data[flag] = False
        elif block_idx[flag] == half_block_size - 1:
            # Append new data to the hdf5
            hdf5s[flag]['data'].resize((hdf5s[flag]['data'].shape[0] + spectrogram_block[flag].shape[0]), axis=0)
            hdf5s[flag]['data'][-spectrogram_block[flag].shape[0]:] = spectrogram_block[flag]

            hdf5s[flag]['label'].resize((hdf5s[flag]['label'].shape[0] + mask_block[flag].shape[0]), axis=0)
            hdf5s[flag]['label'][-mask_block[flag].shape[0]:] = mask_block[flag]
        
        elif num_processed == input_file['data'].shape[0] - 1:
            # if end of data, write remaining to both files
            for flag in (b for b in (True, False) if block_idx[b] != half_block_size - 1):
                hdf5s[flag]['data'].resize((hdf5s[flag]['data'].shape[0] + block_idx[flag]), axis=0)
                hdf5s[flag]['data'][-(block_idx[flag] + 1):] = spectrogram_block[flag][:block_idx[flag]+1]

                hdf5s[flag]['label'].resize((hdf5s[flag]['label'].shape[0] + block_idx[flag]), axis=0)
                hdf5s[flag]['label'][-(block_idx[flag] + 1):] = mask_block[flag][:block_idx[flag]+1]

        block_idx[flag] = (block_idx[flag] + 1) % half_block_size
        num_processed += 1

    # Close files
    input_file.close()
    for _, file in hdf5s.items():
        file.close()

if __name__ == "__main__":
    main()