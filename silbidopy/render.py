import numpy as np
from silbidopy.sigproc import magspec, frame_signal
import wavio
import math

def getSpectrogram(audioFile, frame_time_span = 8, step_time_span = 2, spec_clip_min = 0,
                   spec_clip_max = 6, min_freq = 5000, max_freq = 50000,
                   start_time = 0, end_time=-1):
    '''
    Gets and returns a two-dimensional list in which the values encode a spectrogram.

    :param audioFile: the audio file in .wav format for which a spectrogram is generated.
                      This may either be an audio file of time wavio.Wav or a file name
    :param frame_time_span: ms, length of time for one time window for dft
    :param step_time_span: ms, length of time step for spectrogram
    :param spec_clip_min: log magnitude spectrogram min-max normalization, minimum value
    :param spec_clip_max: log magnitude spectrogram min-max normalization, maximum value
    :param min_freq: Hz, lower bound of frequency for spectrogram
    :param max_freq: Hz, upper bound of frequency for spectrogram
    :param start_time: ms, the beginning of where the audioFile is read
    :param end_time: ms, the end of where the audioFile is read. If end > the length of
                     of the file, then the file is read only to its end.

    :returns: A tuple with both the spectrogram and the time at which the
              spectrogram ended in ms: (spectogram, end_time)
    '''

    freq_resolution = 1000 / frame_time_span

    # Load audio file
    if type(audioFile) == wavio.Wav:
        wav_data = audioFile
    else:
        wav_data = wavio.read(audioFile)

    # I copy this from Pu Li's DeepWhistle implementation
    # in wav2spec.py. I do not know why it is necessary
    if wav_data.sampwidth > 2:
            wav_data.sampwidth /= 2 ** (8 * (wav_data.sampwidth - 2))

    # #
    # Split the wave signal into overlapping frames
    # #

    start_frame = int(start_time / 1000 * wav_data.rate)
    end_frame = int((end_time / 1000 + frame_time_span / 1000 - step_time_span / 1000)* wav_data.rate)

    frame_sample_span = int(math.floor(frame_time_span / 1000 * wav_data.rate))
    step_sample_span = step_time_span / 1000 * wav_data.rate
    # No frames if the audio file is too short
    if wav_data.data[start_frame:end_frame].shape[0] < frame_sample_span:
        frames = []
    else:
        frames = frame_signal(wav_data.data.ravel()[start_frame:end_frame], frame_sample_span, step_sample_span)
    
    # #
    # Make spectrogram
    # #
    NFFT = len(frames[0])

    # Compute magnitude spectra
    singal_magspec = magspec(frames, NFFT)

    # Include only the desired frequency range
    clip_bottom = int(min_freq // freq_resolution)
    clip_top = int(max_freq // freq_resolution) 
    spectogram = singal_magspec.T[clip_bottom:clip_top]
    spectogram = np.log10(spectogram)

    spectogram = normalize3(spectogram, spec_clip_min, spec_clip_max)


    # Flip spectrogram to match expectations for display
    # and DO NOT scale to be 0-255
    spectrogram_flipped = spectogram[::-1, ]

    actual_end_time = start_time + spectrogram_flipped.shape[1] * step_time_span
    return spectrogram_flipped, actual_end_time



def getAnnotationMask(annotations, frame_time_span = 8, step_time_span = 2,
                      min_freq = 5000, max_freq = 50000, start_time = 0, end_time=-1):
    '''
    Gets and returns a two-dimensional list in which the values encode a mask of the annotations.
    The generated mask will have the same shape as will a spectrogram generated by getSpectrogram
    with equivalent parameters.

    :param annotations: The two dimensional array with contours on the first axis and with
                        (time_s,freq_hz) nodes on the second axis. As returned from
                        tonalReader.getTimeFrequencyContours().
    :param frame_time_span: ms, length of time for one time window for dft
    :param step_time_span: ms, length of time step for spectrogram
    :param min_freq: Hz, lower bound of frequency for spectrogram
    :param max_freq: Hz, upper bound of frequency for spectrogram
    :param start_time: ms, the beginning of where the audioFile is read
    :param end_time: ms, the end of where the audioFile is read. -1 reads until the end

    :returns: (annotation mask, positive flag)
    '''

    

    # Get dimensions for mask
    image_width = int((end_time - start_time) / step_time_span)
    image_height = int((max_freq - min_freq) * frame_time_span/1000)

    mask = np.zeros((image_height, image_width))

    # Get only the annotations that will be present in the mask
    low = np.searchsorted([a[-1][0] for a in annotations], start_time / 1000, side='right')
    high = np.searchsorted([a[0][0] for a in annotations], end_time / 1000, side='left')

    annotations = annotations[low:high]

    positive_flag = False

    # if no annotations to plot
    if (low >= high):
         return mask, positive_flag

    freq_resolution = 1000 / frame_time_span
    time_span = (end_time - start_time)

    # plot the portions of annotations that are within the time-frequency range
    for annotation in annotations:
        prev_time_frame = 0
        prev_freq_frame = 0
        first_flag = True
        for time, freq in annotation:

            # get approximate pixel frame for timestamp & frequency
            time_frame = (time*1000 - start_time) * image_width / time_span
            freq_frame = (freq - min_freq) / freq_resolution

            if first_flag:
                prev_time_frame = time_frame
                prev_freq_frame = freq_frame
                first_flag = False
                continue
            
            # Interpolating line function
            freq_time_line = (
                lambda x: freq_frame + (prev_freq_frame - freq_frame) / 
                (prev_time_frame - time_frame)*(x - time_frame) )

            distance = np.sqrt((time_frame-prev_time_frame)**2 + (freq_frame - prev_freq_frame)**2)
            # Draw interpolating line
            for t in list(np.linspace(prev_time_frame, time_frame, math.ceil(distance)+1)):
                
                # check that time is within the image
                if round(t) < 0 or round(t) >= image_width:
                    continue

                # get frequency from interpolation line.
                if time_frame - prev_time_frame < 1e-10:
                    current_freq = freq
                else:
                    current_freq = freq_time_line(t)
                
                # Check that frequency is within the image
                if round(current_freq) < 0 or round(current_freq) >= image_height:
                    continue
                
                # Draw pixel
                mask[round(current_freq), round(t)] = 1
                positive_flag = True

            prev_time_frame = time_frame
            prev_freq_frame = freq_frame
    
    # Flip spectrogram to match expectations for display
    # and DO NOT scale to be 0-255
    mask = mask[::-1, ]

    return mask, positive_flag



####### UTILITY #######

# Credit to Pu Li https://github.com/Paul-LiPu/DeepWhistle
# min-max normalization
def normalize3(mat, min_v, max_v):
    mat = np.clip(mat, min_v, max_v)
    return (mat - min_v) / (max_v - min_v)
