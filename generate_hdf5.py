import h5py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import wavio
import argparse
import numpy as np
import helper_functions as wav2spec
from write_images import write_images
from silbidopy.readBinaries import tonalReader
from silbidopy.render import getSpectrogram, getAnnotationMask



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='the path containing .wav files')
    parser.add_argument('--annotation_dir', type=str, required=True, help='the path containing .bin files')
    parser.add_argument('--output_file', type=str, required=True, help='The file path and name of the output hdf5 file')

    parser.add_argument('--frame_time_span', type=int, default=8, help='ms, length of time for one time window for dft')
    parser.add_argument('--step_time_span', type=int, default=2, help='ms, length of time step for spectrogram')
    parser.add_argument('--spec_clip_min', type=float, default=0, help='log magnitude spectrogram min-max normalization, minimum value')
    parser.add_argument('--spec_clip_max', type=float, default=6, help='log magnitude spectrogram min-max normalization, maximum value')
    parser.add_argument('--min_freq', type=int, default=5000, help='Hz, lower bound of frequency for spectrogram')
    parser.add_argument('--max_freq', type=int, default=50000, help='Hz, upper bound of frequency for spectrogram')
    parser.add_argument('--time_patch_frames', type=int, default=64, help='number of time frames, the length of each datum')
    parser.add_argument('--freq_patch_frames', type=int, default=64, help='number of frequency frames, the height of each datum')
    parser.add_argument('--time_patch_advance', type=int, default=64, help='number of frames, the time distance between patches')
    parser.add_argument('--freq_patch_advance', type=int, default=64, help='number of frames, the frequency distance between patches')
    parser.add_argument('--patches_per_block', type=int, default=128, help='the number of patches computed before each write. Does not effect output, only RAM use during execution')


    config = parser.parse_args()
    ## parameter setting
    frame_time_span = config.frame_time_span # ms, length of time for one time window to do dft.
    step_time_span = config.step_time_span # ms, length of time step.
    spec_clip_min = config.spec_clip_min
    spec_clip_max = config.spec_clip_max # log magnitude spectrogram min-max normalization parameter
    min_freq = config.min_freq # Hz, lower bound of frequency for spectrogram
    max_freq = config.max_freq # Hz, upper bound of frequency for spectrogram

    time_patch_frames = config.time_patch_frames
    freq_patch_frames = config.freq_patch_frames
    time_patch_advance = config.time_patch_advance
    freq_patch_advance = config.freq_patch_advance
    patches_per_block = config.patches_per_block


    # collect all .wav files
    wav_files = wav2spec.find_wav_files(config.audio_dir)

    # collect all .wav filenames
    wav_names = list(map(os.path.basename, wav_files))

    wav_file_dict = {wav_names[i] : wav_files[i] for i in range(len(wav_names))}

    # collect all .bin files.
    exp_group = os.path.basename(config.annotation_dir)
    bin_files = wav2spec.findfiles(config.annotation_dir, fnmatchex='*.bin')

    # find all .wav files that have corresponding .bin files.
    anno_wav_filenames = list(map(wav2spec.bin2wav_filename, bin_files))
    try:
        anno_wav_files = [wav_file_dict[filename] for filename in anno_wav_filenames]
    except KeyError as ex:
        raise Exception(f"Could not find audio file {str(ex)} corresponding to binary file.")
    
    # Useful values
    freq_resolution = 1000 / frame_time_span
    patch_freq_length_hz = freq_resolution * freq_patch_frames
    freq_patch_advance_hz = freq_resolution * freq_patch_advance
    patch_time_length_ms = step_time_span * time_patch_frames
    time_patch_advance_ms = step_time_span * time_patch_advance
    
    

    # Build hdf5 one wav file at a time
    h5f = h5py.File(config.output_file, 'w')
    have_not_added_data = True
    spectrogram_block = np.zeros((patches_per_block, freq_patch_frames, time_patch_frames), dtype="f4")
    mask_block = np.zeros((patches_per_block, freq_patch_frames, time_patch_frames), dtype="f4")
    positive_flag_block = np.zeros((patches_per_block), dtype="f4")
    num_patches_processed = 0
    for i in range(0, len(anno_wav_filenames)):
        print('Processing audio file: %d/%d "%s"' % (i+1, len(anno_wav_filenames), anno_wav_filenames[i]))
        wav_file = anno_wav_files[i]
        wav_filename = os.path.basename(wav_file)
        wav_filename = wav_filename.split('.wav')[0]

        wav = wavio.read(wav_file)

        contours = tonalReader(bin_files[i]).getTimeFrequencyContours()
    
        # Length in ms
        audio_file_length = wav.data.shape[0] / wav.rate * 1000


        # For both times & frequencies, each patch's start & end
        # All patches will be the same size, i.e. the ones near
        # the extremeties that could be smaller are not included
        patches = []
        freq = min_freq
        while freq < max_freq - patch_freq_length_hz:
            time = 0
            while time < audio_file_length - patch_time_length_ms - frame_time_span:
                patches.append((freq, freq + patch_freq_length_hz, time, time + patch_time_length_ms))
                time += time_patch_advance_ms
            freq += freq_patch_advance_hz
     
        # write to hdf5 in groups of patches defined by patches_per_block
        for patch in patches:

            block_idx = num_patches_processed % patches_per_block

            start_freq, end_freq, start_time, end_time = patch

            spectrogram, actual_end = getSpectrogram(wav, frame_time_span=frame_time_span,step_time_span=step_time_span,
                                        spec_clip_min=spec_clip_min, spec_clip_max=spec_clip_max, min_freq=start_freq,
                                        max_freq=end_freq, start_time=start_time, end_time=end_time)
            mask, positive_flag = getAnnotationMask(contours, frame_time_span=frame_time_span,step_time_span=step_time_span,
                min_freq=start_freq, max_freq=end_freq, start_time=start_time, end_time=end_time)

            # Save to block
            spectrogram_block[block_idx] = spectrogram
            mask_block[block_idx] = mask
            positive_flag_block[block_idx] = 1.0 if positive_flag else 0.0

            # Save block of patches to hdf5
            if (block_idx == patches_per_block - 1 or num_patches_processed == len(patches) - 1) and have_not_added_data:
                h5f.create_dataset('data', data=spectrogram_block[:block_idx+1], compression="gzip", chunks=True, maxshape=(None,freq_patch_frames,time_patch_frames))
                h5f.create_dataset('label', data=mask_block[:block_idx+1], compression="gzip", chunks=True, maxshape=(None,freq_patch_frames,time_patch_frames))
                h5f.create_dataset('positive_flag', data=positive_flag_block[:block_idx+1], compression="gzip", chunks=True, maxshape=(None,))
                have_not_added_data = False
            elif block_idx == patches_per_block - 1 or num_patches_processed == len(patches) - 1:
                # Append new data to the hdf5
                h5f['data'].resize((h5f['data'].shape[0] + block_idx+1), axis=0)
                h5f['data'][-(block_idx+1):] = spectrogram_block[:block_idx+1]

                h5f['label'].resize((h5f['label'].shape[0] + block_idx+1), axis=0)
                h5f['label'][-(block_idx+1):] = mask_block[:block_idx+1]

                h5f['positive_flag'].resize((h5f['positive_flag'].shape[0] + block_idx+1), axis=0)
                h5f['positive_flag'][-(block_idx+1):] = positive_flag_block[:block_idx+1]

            num_patches_processed += 1

    h5f.close()

if __name__ == "__main__":
    main()