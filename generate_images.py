import sys

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import helper_functions as wav2spec
from write_images import write_images

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='the path containing .wav files')
    parser.add_argument('--annotation_dir', type=str, required=True, help='the path containing .bin files')
    parser.add_argument('--output_dir', type=str, required=True, help='the path to output images')

    parser.add_argument('--frame_time_span', type=int, default=8, help='ms, length of time for one time window for dft')
    parser.add_argument('--step_time_span', type=int, default=2, help='ms, length of time step for spectrogram')
    parser.add_argument('--spec_clip_min', type=float, default=0, help='log magnitude spectrogram min-max normalization, minimum value')
    parser.add_argument('--spec_clip_max', type=float, default=6, help='log magnitude spectrogram min-max normalization, maximum value')
    parser.add_argument('--min_freq', type=int, default=5000, help='Hz, lower bound of frequency for spectrogram')
    parser.add_argument('--max_freq', type=int, default=50000, help='Hz, upper bound of frequency for spectrogram')
    parser.add_argument('--split_time', type=int, default=3000, help='ms, length of time for each output spectrogram image.')

    config = parser.parse_args()
    ## parameter setting
    frame_time_span = config.frame_time_span # ms, length of time for one time window to do dft.
    step_time_span = config.step_time_span # ms, length of time step.
    clip_min = config.spec_clip_min
    clip_max = config.spec_clip_max # log magnitude spectrogram min-max normalization parameter
    min_freq = config.min_freq # Hz, lower bound of frequency for spectrogram
    max_freq = config.max_freq # Hz, upper bound of frequency for spectrogram
    split_time = config.split_time # ms, length of time for each output spectrogram image.


    # collect all .wav files
    wav_files = wav2spec.find_wav_files(config.audio_dir)

    # collect all .wav filenames
    wav_names = list(map(os.path.basename, wav_files))
    # if len(wav_files) == 1:
    #     wav_names = [wav_names]
    wav_file_dict = {wav_names[i] : wav_files[i] for i in range(len(wav_names))}

    # collect all .bin files.
    exp_group = os.path.basename(config.annotation_dir)
    bin_files = wav2spec.findfiles(config.annotation_dir, fnmatchex='*.bin')

    # find all .wav files that have corresponding .bin files.
    anno_wav_filenames = list(map(wav2spec.bin2wav_filename, bin_files))
    anno_wav_files = [wav_file_dict[filename] for filename in anno_wav_filenames]

    ### set output directory
    imsave_output_dir = config.output_dir
    wav2spec.check_dir(imsave_output_dir)

    # universal normalized magnitute spectrum
    for i in range(0, len(anno_wav_filenames)):
        print('Processing audio file: %d/%d' % (i+1, len(anno_wav_filenames)))
        wav_file = anno_wav_files[i]
        wav_filename = os.path.basename(wav_file)
        wav_filename = wav_filename.split('.wav')[0]

        output_dir = imsave_output_dir + '/' + wav_filename
        wav2spec.check_dir(output_dir)

        count = write_images(anno_wav_files[i], bin_files[i], output_dir, frame_time_span=frame_time_span,step_time_span=step_time_span,
                                     spec_clip_min=clip_min, spec_clip_max=clip_max, min_freq=min_freq,
                                     max_freq=max_freq, split_time = split_time)
        print('number of output: ' + str(count))


if __name__ == "__main__":
    main()