# Music Emotion Recognition Using Advanced Machine Learning to Analyze Lyrics and Audio Features

## Introduction

This project is dedicated to understanding emotional expression in music through a thorough analysis of lyrics and audio features, aiming to develop an advanced model for music emotion analysis that integrates lyrical and audio data. The primary goal of this research is to deepen our understanding of emotional content in music, with a special emphasis on enhancing existing research methods using the MoodyLyrics and MoodyLyrics4Q datasets. The essence of this study lies in the sophisticated combination of complex NLP techniques, Spotify audio features, and machine learning. It demonstrates that integrating Spotify audio features can improve the model's generalization and accuracy. This research not only deepens our understanding of the impact of emotions in music but also provides a valuable and validated data resource, along with future directions for exploring the emotional landscape in contemporary music.

## Project Setup

### Clone Repository

Clone the repository to your local machine:

```bash
git clone https://git.cs.bham.ac.uk/projects-2023-24/yxz1225.git
```
Change directory to the cloned repository:

```bash
cd yxz1225
```

## Research Environment Setup

### Python Version and Key Libraries

The project requires Python 3.10.13; please ensure it's installed, available for download at: 

- Python 3.10.13: [Download Python](https://www.python.org/downloads/)

```bash
pip install python==3.10.13
```

Install the necessary libraries using the `requirements.txt` file included in the repository:

```bash
pip install -r requirements.txt
```

### GPU Support

For GPU support, this project requires CUDA 11.2 and cuDNN 8.1 for TensorFlow 2.10.0:

- CUDA Toolkit 11.2: [Download CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- cuDNN 8.1: [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## Project Structure


### Data Folder
Moodylyrcis, Moodylyrcis4Q, and Top100 datasets, with data analysis and visualization scripts.


- **data_analysis**: Analysis and visualization scripts.

- **data_moody**: 'Moodylyrcis' and 'Moodylyrcis4Q' datasets, plus GloVe and Word2Vec embeddings.
  - **data_pre**: Preprocessing experiment results.

- **data_top100**: Contains raw and predicted data for the 'Top100' dataset, as well as visualization and analysis scripts.
  - **raw**: 'Top100' dataset's raw data and prediction scripts.
  - **predicted**: Post-prediction processed data with visualization and analysis scripts.

### Model Folder
Replication scripts for paper models, enhancements using lyric and audio features, saved models, and testing scripts.

- **model_audio**: Training scripts using audio-only features.

- **model_combine**: Comprehensive model scripts integrating lyrical and audio features.

- **model_improve**: Improvement scripts based on research paper replications for lyrics-only models.

- **model_replication**: Replication scripts for research paper models.

- **model_save**: Stores comprehensive models' states, lyric-focused models' states, tokenizers, label encoders, and Top100 dataset-specific models.
  - **combine**: Comprehensive models' saved states.
  - **lyrics_only**: Lyric features-focused models' saved states.
  - **tokenizer**: Model tokenizers and label encoders.
  - **top100**: Models trained for the Top100 dataset.

- **model_test**: Testing and evaluation scripts.

### Preprocessing Folder
Scripts for lyrics and audio features processing and preprocessing experiments.

- **pre_audio_features**: Extraction and cleaning scripts for audio features.

- **pre_combination_test**: Preprocessing technique experiment scripts.

- **pre_lyrics**: Lyric fetching and cleaning scripts.

### Research Papers Folder
Papers related to the datasets and baseline studies.



## Acknowledgments
- Special thanks to the invaluable data support provided by Genius and Spotify, which significantly contributed to the success of this research.



