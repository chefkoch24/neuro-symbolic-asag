# README
## Requirements
For setup all requirements the following steps has to be done:

`pip install -r requirements.txt`

`python -m spacy download de_core_news_lg`

`python -m spacy download en_core_web_lg`

`python -m nltk.downloader -d /path/to/data/storage/ wordnet`

`python -m nltk.downloader -d /path/to/data/storage/ omw-1.4`

`python -m nltk.downloader -d /path/to/data/storage/ stopwords`

for Slurm cluster: `pip uninstall nvidia_cublas_cu11`

## Configurations
The configurations that are shared between all scripts are set in the file `config.py`.

## Preprocess raw data
For preprocessing the raw data from xml and csv files, run the following command:

`python preprocess_raw_data.py`


## Weak Supervision
The raw data is used in the weak supervision by applying the different labeling functions. 

The first one is based on learning a HMM. The other option is to annotate the data with labeling functions without learning an additional model.

`python weak_supervision_hmm.py`

`python weak_supervision.py`

The labeling functions are evaluated by running

`python evaluate_labeling_functions.py`

## Aggregate Labels
All labels are aggregated by running the following scripts.

For HMM learned model:

`python aggregation_hmm.py`

For labeling functions:

`python aggregation.py`

The aggregation is evaluated by running

`python evaluate_weak_supervision.py`

## Align Labels
because the alignment library can't be exectued on the ML Server, the alignment has to be done locally.
The alignment is done by running the following script:

`python align_labels.py --tokenizer_name tokenizer_name`


## Train Justification Cue Detection Model
Trains the justification cue detection model, 
all configurations are set in the file `config.py`.

`python training_justification_cue_detection.py`

`python training_span_prediction.py`

## Grading
Trains the final grading model

`python grading.py`

