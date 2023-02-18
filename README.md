# README
## Requirements
`python -m spacy download de_core_news_lg`

`python -m spacy download en_core_web_lg`

## Experiments
The experiments run the whole pipleline with the respective configurations.

#TODO: `python experiments.py`


## Configurations
The configurations are stored in the config.py file

## Preprocess raw data
For preprocessing the raw data from xml and csv files, run the following command:

`python preprocess_raw_data.py`

Saves the splitted datasets train and dev data in csv format. 
In addition the rubrics are stored as json file containing
all key elements. 

## Weak Supervision
Annotates the data with the labeling functions

`python weak_supervision.py`

Outputs the annotated data for every labeling function and all 
attributes from the raw data as json file.

## Aggregate Labels
Aggregates the labels from the labeling functions

`python aggregation.py`

Saves the dataset including texts and the final silver labels

## Preprocess Data for Justification Cue Detection Model
Preprocesses the data for the justification cue detection model

`python preprocess_justification_cue_detection.py`

## Train Justification Cue Detection Model
Trains the justification cue detection model

`python training.py`

Evaluation:
    
`python evaluation.py`

## Grading
Trains the final grading model based on 

`python grading.py`

