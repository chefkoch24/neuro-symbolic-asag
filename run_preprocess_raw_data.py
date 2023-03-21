# preprocess raw data

# IMPORTS
import json
import os
import xml.etree.ElementTree as et
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config


# FUNCTIONS
def clean_text(text):
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    return text


def extract_data(path):
    labels, scores, student_answers, tutor_feedbacks, question_ids, reference_answers, languages, questions = [], [], [], [], [], [], [], []
    for folder in os.listdir(path):
        for files in os.listdir(path + '/' + folder):
            if files.endswith('.xml'):
                # if files.endswith(target_question+'.xml'):
                if folder.find('german') != -1:
                    language = 'de'
                else:
                    language = 'en'
                root = et.parse(path + '/' + folder + '/' + files).getroot()
                question_id = os.path.splitext(files)[0]  # get the name of the question as question_id
                question = clean_text(root.find('questionText').text)
                # get reference and student answers from the files
                ref_answers = [x for x in root.find('referenceAnswers')]
                stud_answers = [x for x in root.find('studentAnswers')]
                if len(ref_answers) == 1:
                    ref_answers = clean_text(ref_answers[0].text)
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
                        questions.append(question)

    data = {'question_id': question_ids, 'tutor_feedback': tutor_feedbacks, 'label': labels, 'score': scores,
                'lang': languages, 'student_answer': student_answers , 'question': questions, 'reference_answer':reference_answers}
    return data

def extract_rubrics(path):
    rubric_elements = []
    question_ids = []
    for file in os.listdir(path):
        filename = file.rsplit('.', maxsplit=1)[0]
        question_ids.append(clean_text(filename))
        df = pd.read_csv(path + '/' + filename + ".csv", delimiter=',', header=0)
        rubric_elements.append(df)
    rubrics = {}
    for qi, r in zip(question_ids, rubric_elements):
        r['key_element'] = clean_text(r['key_element'].str)
        r['points'] = [float(i) for i in r['points']]
        rubrics[qi] = r
    return rubrics

def main():
    config = Config()
    train_data = extract_data(config.PATH_RAW_DATA + '/training')
    test_data = extract_data(config.PATH_RAW_DATA + '/unseen_answers')
    rubrics = extract_rubrics(config.PATH_RAW_RUBRIC)
    X = pd.DataFrame(data=train_data)
    y = X['score']
    X_test = pd.DataFrame(data=test_data)


    X_train, X_dev, y_train, y_dev = train_test_split(X.index, y.index, test_size=0.2, random_state=config.SEED,
                                                      shuffle=True)
    X_train = X.loc[X_train]  # return dataframe train
    X_dev = X.loc[X_dev]
    y_train = y.loc[y_train]
    y_dev = y.loc[y_dev]
    y_train = [float(y) for y in y_train.tolist()]
    y_dev = [float(y) for y in y_dev.tolist()]
    X_train['score'] = y_train
    X_dev['score'] = y_dev
    X_test['score'] = [float(y) for y in X_test['score'].tolist()]
    # X_train.to_json('training_dataset.json')
    records = X_train.to_dict('records')
    # save the list of dictionaries as a JSON file
    with open(config.PATH_DATA + '/training_dataset.json', 'w') as f:
        f.write(json.dumps(records))
    records = X_dev.to_dict('records')
    # save the list of dictionaries as a JSON file
    with open(config.PATH_DATA + '/dev_dataset.json', 'w') as f:
        f.write(json.dumps(records))
    # Save the rubric
    new_data = dict()
    for k in rubrics.keys():
        new_data[k] = rubrics[k].to_dict()
    with open(config.PATH_DATA + '/' + 'rubrics.json', 'w') as fout:
        json.dump(new_data, fout)

    # Save the test data
    records = X_test.to_dict('records')
    with open(config.PATH_DATA + '/test_dataset.json', 'w') as f:
        f.write(json.dumps(records))

if __name__ == '__main__':
    main()