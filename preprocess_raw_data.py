# preprocess raw data

# IMPORTS
import json
import os
import xml.etree.ElementTree as et
import pandas as pd
from sklearn.model_selection import train_test_split
import config


# FUNCTIONS
def clean_text(text):
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    return text


def get_additional_data(data, dictonary):
    ref_answers = []
    for i, e in data.iterrows():
        qid = e['question_id']
        reference_answer = dictonary[qid]
        ref_answers.append(clean_text(reference_answer))
    return ref_answers


def save_to_csv(X_train, X_dev, path):
    sep = "\t"  # custom seperator needed that it not make troubles while reading in the data
    directory = path
    if not os.path.exists(directory):
        os.mkdir(directory)
    X_train.to_csv(path + '/' + "x_train.csv", sep=sep)
    X_dev.to_csv(path + '/' + "x_dev.csv", sep=sep)
    print('successfully saved')


def save_as_json(data, path, file_name):
    new_data = dict()
    for k in data.keys():
        new_data[k] = data[k].to_dict()
    with open(path + '/' + file_name, 'w') as fout:
        json.dump(new_data, fout)
        print('saved', file_name)


def remove_empty_responses(dataframe):
    for i, a in dataframe.iterrows():
        if len(a['student_answer']) <= 1:
            dataframe.drop(i, axis=0, inplace=True)
            print('repsonse removed')


# SCRIPT
labels, scores, student_answers, tutor_feedbacks, question_ids, reference_answers, languages = [], [], [], [], [], [], []
for folder in os.listdir(config.PATH_RAW_DATA):
    for files in os.listdir(config.PATH_RAW_DATA + '/' + folder):
        if files.endswith('.xml'):
            # if files.endswith(target_question+'.xml'):
            if folder.find('german') != -1:
                language = 'de'
            else:
                language = 'en'
            root = et.parse(config.PATH_RAW_DATA + '/' + folder + '/' + files).getroot()
            question_id = os.path.splitext(files)[0]  # get the name of the question as question_id
            question = clean_text(root.find('questionText').text)
            # get reference and student answers from the files
            ref_answers = [x for x in root.find('referenceAnswers')]
            stud_answers = [x for x in root.find('studentAnswers')]
            if len(ref_answers) == 1:
                ref = clean_text(ref_answers[0].text)
                for x in stud_answers:
                    # arrange the text sequences according to the set parameters
                    response = clean_text(x.find('response').text)
                    tutor_feedback = clean_text(x.find('response_feedback').text)
                    score = clean_text(x.find('score').text)
                    label = clean_text(x.find('verification_feedback').text)
                    if label == "Correct":
                        label = 'CORRECT'
                    elif label == "Partially correct":
                        label = 'PARTIAL_CORRECT'
                    elif label == "Incorrect":
                        label = 'INCORRECT'

                    # lowercase data
                    reference_answers.append(ref_answers)
                    languages.append(language)
                    question_ids.append(question_id)
                    student_answers.append(response)
                    labels.append(label)
                    scores.append(score)
                    tutor_feedbacks.append(tutor_feedback)
data = {'question_id': question_ids, 'tutor_feedback': tutor_feedbacks, 'label': labels, 'score': scores, 'lang': languages, 'student_answer': student_answers}

answer_data = pd.DataFrame(data=data)

rubric_elements = []
question_ids = []
for file in os.listdir(config.PATH_RAW_RUBRIC):
    filename = file.rsplit('.', maxsplit=1)[0]
    question_ids.append(clean_text(filename))
    df = pd.read_csv(config.PATH_RAW_RUBRIC + '/' + filename + ".csv", delimiter=',', header=0)
    rubric_elements.append(df)

rubrics = {}
for qi, r in zip(question_ids, rubric_elements):
    r['key_element'] = clean_text(r['key_element'].str)
    r['points'] = [float(i) for i in r['points']]
    rubrics[qi] = r

reference_answers = {}
questions = {}
for folder in os.listdir(config.PATH_RAW_DATA):
    for files in os.listdir(config.PATH_RAW_DATA + "/" + folder):
        if files.endswith('.xml'):
            # if files.endswith(target_question+'.xml'):
            root = et.parse(config.PATH_RAW_DATA + '/' + folder + '/' + files).getroot()
            question_id = files.rsplit('.', maxsplit=1)[0]
            # get reference and student answers from the files
            ref_answers = [x for x in root.find('referenceAnswers')]
            reference_answers[question_id] = clean_text(ref_answers[0].text)
            questions[question_id] = clean_text(root.find('questionText').text)

y = answer_data['score']
X = answer_data

X_train, X_dev, y_train, y_dev = train_test_split(X.index, y.index, test_size=0.2, random_state=config.SEED, shuffle=True)
X_train = X.loc[X_train]  # return dataframe train
X_dev = X.loc[X_dev]
y_train = y.loc[y_train]
y_dev = y.loc[y_dev]
y_train = y_train.tolist()
y_train = [float(y) for y in y_train]
y_dev = y_dev.tolist()
y_dev = [float(y) for y in y_dev]
X_train['score'] = y_train
X_dev['score'] = y_dev

X_train['reference_answer'] = get_additional_data(X_train, reference_answers)
X_dev['reference_answer'] = get_additional_data(X_dev, reference_answers)
X_train['question'] = get_additional_data(X_train, questions)
X_dev['question'] = get_additional_data(X_dev, questions)

remove_empty_responses(X_train)
remove_empty_responses(X_dev)

#X_train.to_json('training_dataset.json')
records = X_train.to_dict('records')
# save the list of dictionaries as a JSON file
with open(config.PATH_DATA + '/training_dataset.json', 'w') as f:
    f.write(json.dumps(records))
records = X_dev.to_dict('records')
# save the list of dictionaries as a JSON file
with open(config.PATH_DATA + '/dev_dataset.json', 'w') as f:
    f.write(json.dumps(records))
save_as_json(rubrics, config.PATH_DATA, 'rubrics.json')