# This script finally grades the student answers
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = 'chefkoch24/justification-cue-tagging'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# TODO if I know if iterative prediction or combined is better
