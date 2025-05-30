
import torch
import torch.nn.functional as F

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

class Predictor:
    def calculate_perplexity(logits, target_ids):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        return torch.exp(loss)

    def calculate_bleu(reference, hypothesis):
        return sentence_bleu([reference.split()], hypothesis.split())

    def calculate_rouge(reference, hypothesis):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]

    def calculate_accuracy(predictions, ground_truths):
        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        return correct / len(predictions) if predictions else 0.0
