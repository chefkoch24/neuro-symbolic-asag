from config import Config
from prediction import PredictSpan, PredictToken

config = Config(task='span_prediction',
                test_file='dev_dataset_aligned_labels_SpanBERT_spanbert-base-cased.json',
                checkpoint_path='logs/span_prediction_SpanBERT_spanbert-base-cased_bs-8/version_0/checkpoints/checkpoint-epoch=00-val_loss=6.24.ckpt',
                model='SPANBERT/spanbert-base-cased',
                )
predict_span = PredictSpan(config)
predict_span.predict()

config = Config(task='token_classification',
                    test_file='dev_dataset_aligned_labels_SpanBERT_spanbert-base-cased.json',
                    checkpoint_path='logs/span_prediction_SpanBERT_spanbert-base-cased_bs-8/version_0/checkpoints/checkpoint-epoch=00-val_loss=6.24.ckpt',
                    model='SPANBERT/spanbert-base-cased',
                    context=True
                    )
PredictToken(config).predict()

