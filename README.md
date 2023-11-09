# DeepWhistleGeneration
A simple utilitiy, based on [DeepWhistle](https://github.com/Paul-LiPu/DeepWhistle)'s generator, to generate image data from audio files and *[silbido](https://github.com/MarineBioAcousticsRC/silbido)* annotation files.

The interface is the same as [DeepWhistle](https://github.com/Paul-LiPu/DeepWhistle)'s. The difference here is in the manner whereby the annotations are drawn to the golden label images. NOTE, the difference means that this version was NOT used in [Learning Deep Models from Synthetic Data for Extracting Dolphin Whistle Contours](https://arxiv.org/abs/2005.08894).

This makes use of [silbidopy](https://github.com/joshua-zingale/silbidopy), which is included in this repository.

You can call the data generator with the following:
```bash
python generate_traindata.py --audio_dir PATH_TO_AUDIO_FILES  \ 
  --annotation_dir PATH_TO_ANNOTATION_FILES --output_dir PATH_TO_OUTPUT_SPECTROGRAM
```
There are also more parameters that may be inspected with
```bash
python generate_traindata.py -h
```
