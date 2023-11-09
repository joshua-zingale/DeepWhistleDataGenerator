import wavio
from silbidopy.render import getSpectrogram, getAnnotationMask
from silbidopy.readBinaries import tonalReader
from PIL import Image
def write_images(audio_filename, binary_filename, output_dir, frame_time_span = 8, step_time_span = 2,
                 spec_clip_min = 0, spec_clip_max = 6, min_freq = 5000, max_freq = 50000,
                 split_time = 3000):
    
    wav= wavio.read(audio_filename)
    contours = tonalReader(binary_filename).getTimeFrequencyContours()
    
    # Length in ms
    audio_file_length = wav.data.shape[0] / wav.rate * 1000

    num_images = 0
    # write images
    time = 0
    while time < audio_file_length:
        end_time = min(time + split_time, audio_file_length)

        spectrogram, actual_end = getSpectrogram(wav, frame_time_span=frame_time_span,step_time_span=step_time_span,
                                     spec_clip_min=spec_clip_min, spec_clip_max=spec_clip_max, min_freq=min_freq,
                                     max_freq=max_freq, start_time=time, end_time=end_time)
        mask = getAnnotationMask(contours, frame_time_span=frame_time_span,step_time_span=step_time_span,
                                 min_freq=min_freq, max_freq=max_freq, start_time=time, end_time=actual_end)
        
        spec_im = Image.fromarray(spectrogram).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=8)
        mask_im = Image.fromarray(mask).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=8)

        spec_im.save(output_dir + f"/{num_images}-a.png")
        mask_im.save(output_dir + f"/{num_images}-b.png")

        num_images += 1

        time = end_time


    return num_images



