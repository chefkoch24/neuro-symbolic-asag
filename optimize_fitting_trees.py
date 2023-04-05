import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report
from config import Config
import myutils as utils

def _init_symbolic_models(mode, rubrics):
        symbolic_models = {}
        for qid in list(rubrics.keys()):
            max_features = len(rubrics[qid]['key_element'].tolist())
            if mode == 'classification':
                if MAX_DEPTH == None:
                    symbolic_models[qid] = GradientBoostingClassifier()
                else:
                    symbolic_models[qid] = GradientBoostingClassifier()#DecisionTreeClassifier(max_features=max_features, max_depth=max_features)
        return symbolic_models

config = Config(
    train_file='data/training_dataset.json',
    dev_file='data/dev_dataset.json',
)
#MATCHING = 'exact'
TASK = 'token_classification'
#TASK = 'span_prediction'
MATCHING = 'fuzzy'
LR = 0.1
MODE = 'classification'
#MAX_DEPTH = None
MAX_DEPTH ='rubric_length'
MAX_FEATURES = 'rubric_length'
rubrics = utils.load_rubrics(config.PATH_RUBRIC)
PATH_TO_TRAIN_SCORING_VECTORS = 'logs/grading_token_classification_2023-04-03_04-16/scoring_vectors/train_scoring_vectors_epoch_2.json'
PATH_TO_DEV_SCORING_VECTORS = 'logs/grading_token_classification_2023-04-03_04-16/scoring_vectors/test_scoring_vectors_epoch_2.json'
#PATH_TO_TRAIN_SCORING_VECTORS = 'logs/grading_span_prediction_classification_decision_tree/scoring_vectors/train_scoring_vectors_epoch_1.json'
#PATH_TO_DEV_SCORING_VECTORS = 'logs/grading_span_prediction_classification_decision_tree/scoring_vectors/test_scoring_vectors_epoch_1.json'
train_scoring_vectors = utils.load_json(PATH_TO_TRAIN_SCORING_VECTORS)
dev_scoring_vectors = utils.load_json(PATH_TO_DEV_SCORING_VECTORS)
training_data = utils.load_json(config.TRAIN_FILE)
dev_data = utils.load_json(config.DEV_FILE)
hyperparams ={
    'max_features': MAX_FEATURES,
    'max_depth': MAX_DEPTH,
    'matching': MATCHING,
    'mode': MODE,
    'lr': LR,
}
symbolic_models = _init_symbolic_models(MODE, rubrics)
label = 'label' if MODE == 'classification' else 'score'


# Fitting
for qid in list(rubrics.keys()):
    X = train_scoring_vectors[qid]
    y_true = [d[label] for d in training_data if d['question_id'] == qid]
    symbolic_models[qid].fit(np.array(X), np.array(y_true))

# Scoring
scores = {}
cms = {}
reports = {}
for qid in list(rubrics.keys()):
    X = dev_scoring_vectors[qid]
    y_true = [d[label] for d in dev_data if d['question_id'] == qid]
    s = symbolic_models[qid].score(X, y_true)
    scores[qid] = s
    y_pred = symbolic_models[qid].predict(X)
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    reports[qid] = report
    cms[qid] = cm

final_score = sum(scores.values())/len(scores)
scores['final_score'] = final_score

experiment_name = utils.get_experiment_name(['decision_tree_optimization',TASK, MODE, MATCHING, MAX_DEPTH])
print(experiment_name, scores)
print(cms)
print(reports)
#utils.save_json(hyperparams | scores, 'results',file_name=experiment_name + '.json')
