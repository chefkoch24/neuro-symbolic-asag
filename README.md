# README

## Folder Structure

- **corpora** - contains spacy corpora from HMMs
- **data** - contains all pre-processed data files
  - **aggregated** - all files with aggregated labels from weak supervision
- **input** - contains all raw data files, including the rubric
  - **rubrics** - contains all rubric files
  - **safdataset** - contains all safdataset files downloaded from https://github.com/SebOchs/SAF
- **notebooks** - contains all notebooks for analysis and visualization
- **results** - contains all result files. Subfolders for each final experiment
- **logs** - contains all logs while training

## Code Structure

All scripts are in the main folder.
The scripts for running configurations contain the prefix *run*.
Helper scripts contain the prefix *_*.
All other scripts contain the classes according to their name.
The configurations must be adapted in the code for running own experiments.

## Data Splits 

We created different Github branches per data split for the experiments.

- **master** - contains data & setup for German and English questions, including training data from the
unseen questions are used to train the models.
- **monolingual-de** - contains data & setup for German questions
- **monolingual-en** - contains data & setup for English questions
- **multilingual** - contains data & setup for German and English questions, excluding training data from the unseen questions

## Requirements
Python 3.9

To set up all requirements, the following steps have to be done:

`pip install -r requirements.txt`

`python -m spacy download de_core_news_lg`

`python -m spacy download en_core_web_lg`

`python -m nltk.downloader -d /path/to/data/storage/ wordnet`

`python -m nltk.downloader -d /path/to/data/storage/ stopwords`


## Configurations
The configurations of all shared data paths are set in the file `config.py`. To customize paths, you have to adapt this file.

## Preprocess raw data
For preprocessing the raw data from xml and csv files, run the following script:

`python run_preprocess_raw_data.py`

It incorporates all data from the input/safdataset folder and the input/rubrics folder.

If we want to incorporate the unseen questions we run the following script:

`python run_split_unseen_question_dataset.py`

It's necessary to manual move the data to the unseen answers folder.


## Weak Supervision
The raw data is used in the weak supervision by applying the different labeling functions. 

Run all the experiments:

`python run_weak_supervision_experiments.py`

Setup only the final HMM by setting the configuration: 

`python run_weak_supervision.py`


The labeling functions are evaluated by running

`python run_evaluate_labeling_functions.py`


## Aggregate Labels
All labels are aggregated by running the following scripts.

Configure the aggregation procedure.

`python run_aggregation.py`

The aggregation is evaluated by running

`python run_evaluate_weak_supervision.py`

## Align Labels
because the alignment library can't be exectued on the ML Server, the alignment has to be done locally.
The alignment is done by running the following script:

`python run_align_labels.py`


## Justification Cue Detection Model
Trains the justification cue detection model, 
all configurations are set in the file `config.py`.

`python run_experiments_justification_cue_detection.py`

Evaluation

`python run_evaluate_justification_cue_detection.py`

Generates prediction files for the justification cue detection model

`python run_justification_cue_predictions.py`

## Grading
Trains the final grading experiments on the dev set.

`python run_grading_experiments.py`

Train the final models based on the decisions made from the dev set.

`python run_final_grading.py`

For the final evaluation and generation of prediction files run the following script:

`python run_final_eval.py`

it's necessary to link/ uncomment the respective lines to configure the correct model combinations.
All grading models are stored in separate folders in the *logs* folder and contain a hyperparameter file and metrics.csv
which was created during training.

## Evaluation

The final metrics are based on generated prediction files which allow to exclude the unseen questions from the final metrics.
The final results are manually extracted from the notebook `analyze_final_results.ipynb` in the *notebooks* folder.
The respective table is stored in the file `final_results_grading.csv` and `final_results_scoring.csv` in the *results* folder.

## Analysis

In the folder, *notebooks* are all notebooks for deeper analysis and visualizations of the data and results.