# README
## Requirements
For setup all requirements the following steps has to be done:

`pip install -r requirements.txt`

`python -m spacy download de_core_news_lg`

`python -m spacy download en_core_web_lg`

`python -m nltk.downloader -d /path/to/data/storage/ wordnet`

`python -m nltk.downloader -d /path/to/data/storage/ omw-1.4`

`python -m nltk.downloader -d /path/to/data/storage/ stopwords`


## Configurations
The configurations that are shared between all scripts are set in the file `config.py`.

## Preprocess raw data
For preprocessing the raw data from xml and csv files, run the following command:

`python preprocess_raw_data.py`

Saves the splitted datasets train and dev data in json format. 
In addition the rubrics are stored as json file containing
all key elements. 

## Weak Supervision
I have two options for weak supervision. 

The first one is based on learning a HMM. The other option is to annotate the data with labeling functions without learning an additional model.

`python weak_supervision_hmm.py`

`python weak_supervision.py`

Outputs the annotated data for every labeling function and all 
attributes from the raw data as json file.

## Aggregate Labels
For HMM learned model:

`python aggregation_hmm.py`

For labeling functions:

`python aggregation.py`

## Analyze Data
The script outputs the evaluation results for the different weak supervision methods:

`python analyze_aggregation.py`

## Preprocess Data for Justification Cue Detection Model

Needed because the alignment library is implemented in Rust and does not work on the ML Server.
There exists two scripts for preprocessing the data. One for the combined predcition of all justification cues and
one for the prediction of each cue iterativley.

Combined:
The reference answer is added as context with the context flag set to True.

`python preprocess_justification_cue_detection.py --model name_of_model --context True_or_False --train_file train_file --dev_file dev_file`

Iterative:
`python preprocess_iterative_justification_cue_detection.py --model name_of_model --train_file train_file --dev_file dev_file`

## Train Justification Cue Detection Model
Trains the justification cue detection model

Combined (important to add the correct preprocessed file):
`python training_justification_cue_detection.py \
--model model_name \
--context True_or_False \
--train_file train_file \
--dev_file  dev_file`

Iterative:
`python iterative_prediction.py \
--model model_name \
--train_file train_file \
--dev_file  dev_file`

## Grading
Trains the final grading model

`python grading.py`

