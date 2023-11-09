from render import *
from readBinaries import *
from matplotlib import pyplot as plt
from PIL import Image
af = "C:/Users/jzingale8274/Documents/testml/audio/20170515_170000.wav"

ann = "C:/Users/jzingale8274/Documents/testml/annotations/20170515_170000.bin"
tr = tonalReader(ann)

contours = tr.getTimeFrequencyContours()

START = 1
END = 1

o = getSpectrogram(af, frame_time_span=1024, min_freq = 10, max_freq = 120, step_time_span = 25, end_time = 3000)

o2 = getAnnotationMask(contours, frame_time_span=1024, min_freq = 10, max_freq = 120, step_time_span = 25, end_time = 3000)

im = Image.fromarray(o2)
im_2 = im.convert('RGB')
im_2.save("./im.png")

# plt.imshow(o2)
# plt.show()