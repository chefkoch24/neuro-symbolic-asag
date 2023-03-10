# runs all experiments for the justification cue detection task
import subprocess
import logging
import config

for bs in [8, 16, 32]:
    for model in ["distilbert-base-multilingual-cased", "spanbert-base-cased"]:
        # TOKEN CLASSIFICATION
        for context in [True, False]:
            config.MODEL_NAME = model
            config.CONTEXT = context
            config.BATCH_SIZE = bs
            config.TRAIN_FILE = 'training_dataset' + config.MODEL_NAME + '.json'
            config.DEV_FILE = 'dev_dataset' + config.MODEL_NAME + '.json'
            logging.info("Running experiment: " + config.MODEL_NAME + " context: " + str(config.CONTEXT) + " batch size: " + str(config.BATCH_SIZE))
            process = subprocess.Popen(['python', 'training_justification_cue_detection.py'], subprocess.PIPE)
            process.wait()

        # SPAN PREDICTION
        config.TRAIN_FILE = "training_dataset_span_prediction_" + config.MODEL_NAME + ".json"
        config.DEV_FILE = "dev_dataset_span_prediction_" + config.MODEL_NAME + ".json"
        logging.info(
            "Running experiment: " + config.MODEL_NAME + " batch size: " + str(config.BATCH_SIZE))
        process = subprocess.Popen(['python', 'training_span_prediction.py'], subprocess.PIPE)
        process.wait()