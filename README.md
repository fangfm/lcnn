# LCNN
A TensorFlow implementation of light convolutional neural network (LCNN or light CNN).<br>
This implementation is aimed to classify between genuine speech (spoken by human) and playback speech (played by loud speaker). LCNN-based playback speech detection was firstly introduced by [1] and it has been served as a playback spoofing countermeasure to evaluate a "speech-enhancement-based" playback attack method [2].


# Usage
## Train
```bash
./main.py \
	  --train_genuine training_data_genuine_file_list.txt \   # all data should be saved as binary format with float type
	  --train_spoof training_data_spoof_file_list.txt \
	  --dev_genuine dev_data_genuine_file_list.txt \
	  --dev_spoof dev_data_spoof_file_list.txt \
	  --epoch  9 \
	  --batch_size 64 \
	  --lr 0.0001 \        # initial learning rate
	  --dlr 0.9 \          # if classification error rate increases, learning rate will be decreased by this rate
	  --keep_prob 0.5      # dropout
```
<br><br>
Each line of "*file_list.txt" contains a path to a corresponding data file. A data file contains a sequence of spectrogram (see [1] and [2] for details).
"*genuine_file_list.txt" means file list of genuine speech and "\*spoof_file_list.txt" means file list of playback speech.
"dev_data*" means development dataset used for monitoring classification error rate.

## Test
```bash
./main.py \
	  --test_data test_data_file_list.txt \
	  --phase test \
	  --test_dir output_directory # all output is saved in binary format with float type
```

The output file contains a sequence of 2-D vectors and each vector is composed by two probabilities (genuine and playback).

## Reference
[1] Lavrentyeva, Galina, et al. "Audio replay attack detection with deep learning frameworks." Proc. Interspeech. 2017.<br>
[2] F. Fang et al., "Transforming acoustic characteristics to deceive playback spoofing countermeasures of speaker verification systems," IEEE International Workshop on Information Forensics and Security (WIFS), 2018.
