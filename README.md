# FEDA-MedAlarms: Medical Alarm Sound Event Detection

## Overview

FEDA-MedAlarms is a sound event detection (SED) system specifically designed for medical alarm recognition in healthcare environments. The system uses Convolutional Recurrent Neural Networks (CRNN) with mel-spectrogram features to detect and classify medical alarm sounds.

## Features

- **Medical Alarm Classification**: Detects 7 classes of medical alarms:
  - Syringe pump
  - Infusion pump
  - External feeding pump
  - Patient monitor
  - Chest drainage
  - Foot pump
  - Mechanical ventilator

- **Deep Learning Models**: Multiple neural network architectures:
  - CNN3-BiSRNN (3-layer CNN with Bidirectional Simple RNN)
  - CNN3-BiGRU (3-layer CNN with Bidirectional GRU)
  - CNN4-BiGRU (4-layer CNN with Bidirectional GRU)
  - CNN4-stride1x2-BiGRU (4-layer CNN with custom stride)

- **Audio Processing**: 
  - Mel-spectrogram feature extraction
  - Data augmentation with white noise
  - SpecAugment for improved generalization

## Project Structure

```
FEDA-MedAlarms/
├── 01-feature-class7-mel-white.ipynb    # Feature extraction with noise augmentation
├── 02-sed-class7-mel-CNN3-BiGRU.ipynb   # CNN3-BiGRU model training
├── 02-sed-class7-mel-CNN3-BiSRNN.ipynb  # CNN3-BiSRNN model training
├── 02-sed-class7-mel-CNN4-BiGRU.ipynb   # CNN4-BiGRU model training
├── 02-sed-class7-mel-CNN4-stride1x2-BiGRU.ipynb  # CNN4 with custom stride
├── 03-sed-pred-class7-mel_meansd.py     # Model inference and prediction
├── 04-sed-eval_meansd.py                # Performance evaluation
├── utils.py                             # Utility functions
├── metrics.py                           # Evaluation metrics
└── README.md                            # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.4+
- NumPy
- Librosa
- SciPy
- Matplotlib
- Pandas
- scikit-learn
- sed_eval
- dcase_util

For detailed requirements, see `requirements.txt`.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/FEDA-MedAlarms.git
cd FEDA-MedAlarms
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p feat models result_fold eval_result_fold
```

## Usage

### 1. Feature Extraction

Extract mel-spectrogram features with data augmentation:

```bash
jupyter notebook 01-feature-class7-mel-white.ipynb
```

This notebook:
- Loads audio files and annotation data
- Extracts mel-spectrogram features (40 mel bands)
- Applies white noise augmentation at various SNR levels
- Saves processed features for training

### 2. Model Training

Train different CRNN architectures:

```bash
# CNN3-BiGRU model
jupyter notebook 02-sed-class7-mel-CNN3-BiGRU.ipynb

# CNN3-BiSRNN model  
jupyter notebook 02-sed-class7-mel-CNN3-BiSRNN.ipynb

# CNN4-BiGRU model
jupyter notebook 02-sed-class7-mel-CNN4-BiGRU.ipynb

# CNN4 with stride 1x2
jupyter notebook 02-sed-class7-mel-CNN4-stride1x2-BiGRU.ipynb
```

Each training notebook includes:
- Model architecture definition
- SpecAugment data augmentation
- 5-fold cross-validation
- Performance monitoring and early stopping

### 3. Model Inference

Run inference on test data:

```bash
python 03-sed-pred-class7-mel_meansd.py
```

This script:
- Loads trained models from all folds
- Performs ensemble prediction (mean across folds)
- Outputs detection results in time-stamped format
- Calculates performance metrics

### 4. Performance Evaluation

Evaluate model performance using sed_eval:

```bash
python 04-sed-eval_meansd.py
```

This script:
- Computes segment-based and event-based metrics
- Generates class-wise performance statistics
- Outputs results with mean ± standard deviation across folds

## Model Architecture

### CRNN Architecture
- **Input**: Mel-spectrogram (40 mel bands × time frames)
- **CNN Layers**: 3-4 convolutional layers with batch normalization
- **RNN Layers**: Bidirectional GRU/SimpleRNN layers
- **Output**: Multi-label classification with sigmoid activation

### Training Parameters
- **Batch Size**: 480
- **Sequence Length**: 256 frames
- **Learning Rate**: Adam optimizer
- **Data Augmentation**: SpecAugment + white noise
- **Cross-validation**: 5-fold

## Performance Metrics

The system is evaluated using:
- **Segment-based metrics**: F1-score, Precision, Recall, Error Rate
- **Event-based metrics**: F1-score, Precision, Recall, Error Rate
- **Time resolution**: 0.1 seconds
- **Collar tolerance**: 2.0 seconds for event-based evaluation

## Data Format

### Input Audio
- **Sample Rate**: 48 kHz
- **Format**: WAV files
- **Channels**: Mono

### Annotation Format
Tab-separated values with columns:
```
filename    onset_time    offset_time    class_label
```

### Output Format
Detection results in tab-separated format:
```
onset_time    offset_time    class_label
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{DetectionPolyphonicAlarm,
	title = {Detection of {Polyphonic} {Alarm} {Sounds} from {Medical} {Devices} {Using} {Frequency}-{Enhanced} {Deep} {Learning}: {Simulation} {Study}},
	shorttitle = {Detection of {Polyphonic} {Alarm} {Sounds} from {Medical} {Devices} {Using} {Frequency}-{Enhanced} {Deep} {Learning}},
	url = {https://preprints.jmir.org/preprint/35987},
	abstract = {Journal of Medical Internet Research - International Scientific Journal for Medical Research, Information and Communication on the Internet},
	language = {en},
	urldate = {2025-10-26},
	journal = {JMIR Preprints},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on DCASE sound event detection frameworks
- Uses sed_eval library for evaluation metrics
- Incorporates SpecAugment for data augmentation