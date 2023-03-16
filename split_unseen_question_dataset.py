import os
import xml.etree.ElementTree as et
import random
from config import Config


def get_split_points(path, test_size=0.15, seed=42):
    random.seed(seed)
    for folder in os.listdir(path):
        for files in os.listdir(path + '/' + folder):
            if files.endswith('.xml'):
                root = et.parse(path + '/' + folder + '/' + files).getroot()
                student_answers = [x for x in root.find('studentAnswers')]
                length_answers = len(student_answers)
                split_id = int(length_answers * test_size)
                # Split the student answers into two parts
                random.shuffle(student_answers)
                part1 = student_answers[split_id:]
                part2 = student_answers[:split_id]
                question_id = os.path.splitext(files)[0]
                # Create two new XML files with the two parts of the student answers
                filename1 = 'input/tmp/' + question_id + '_training.xml'
                filename2 = 'input/tmp/' + question_id + '_test.xml'
                # Create the root element for the new XML files
                root1 = et.Element('root')
                root2 = et.Element('root')
                # Add the parts to the new root elements
                for item in part1:
                    root1.append(item)
                for item in part2:
                    root2.append(item)

                # Write the new XML files to disk
                et.ElementTree(root1).write(filename1, encoding='utf-8', xml_declaration=True)
                et.ElementTree(root2).write(filename2, encoding='utf-8', xml_declaration=True)

config = Config()
get_split_points('input/safdataset/unseen_questions', test_size=0.15, seed=config.SEED)