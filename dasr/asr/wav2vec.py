import re

import torch
import torch.nn.functional as F
from evaluate import load
from torchmetrics.text import CharErrorRate, WordErrorRate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class TruncatedWER(WordErrorRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds, target, trunc_threshold=1):
        orig_wer = super().__call__(preds, target)
        trunc_wer = orig_wer if orig_wer != float("inf") else trunc_threshold
        return trunc_wer


class TruncatedCER(CharErrorRate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds, target, trunc_threshold=1):
        orig_cer = super().__call__(preds, target)
        trunc_cer = orig_cer if orig_cer != float("inf") else trunc_threshold
        return trunc_cer


class Wav2VecEnv:
    def __init__(
        self,
        device="cuda",
        path_model="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        asr_metric="wer",
        baseline=False,
    ):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(path_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(path_model)

        self.model = self.model.to(device)
        self.model.eval()
        self.freeze_model()

        # self.wer = load("wer")
        # self.cer = load("cer")

        self.wer = TruncatedWER()
        self.cer = TruncatedCER()

        if asr_metric == "wer":
            self.asr_metric = self.wer
        elif asr_metric == "cer":
            self.asr_metric = self.cer
        self.baseline = baseline

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def inference_with_grad(self, speech, attention_mask=None):
        output = self.model(speech, attention_mask=attention_mask)
        tokens_logits = output.logits
        predicted_ids = torch.argmax(tokens_logits, dim=-1)
        pred_texts = self.processor.batch_decode(predicted_ids)
        logprobs = tokens_logits.softmax(dim=2).max(dim=2).values.log().sum(dim=1)
        return pred_texts, logprobs

    def inference_without_grad(self, speech, attention_mask=None):
        with torch.no_grad():
            output = self.model(speech, attention_mask=attention_mask)

        tokens_logits = output.logits
        predicted_ids = torch.argmax(tokens_logits, dim=-1)
        pred_texts = self.processor.batch_decode(predicted_ids)

        return pred_texts

    def get_loss(self, speech, denoisy_speech, noisy_speech=None, gt_transcript=None):
        reference_transcript = self.inference_without_grad(speech)
        denoisy_transcript, logprob = self.inference_with_grad(denoisy_speech)
        reward = [
            -self.asr_metric(preds=[p], target=[r])
            for p, r in zip(denoisy_transcript, reference_transcript)
        ]
        if self.baseline:
            reward = [i + self.baseline for i in reward]
        reward = torch.Tensor(reward).to(self.device)
        loss = -logprob * reward

        if noisy_speech is not None:
            noisy_transcript = self.inference_without_grad(noisy_speech)
        else:
            noisy_transcript = None

        stats = self.get_stats(
            loss,
            reward,
            logprob,
            reference_transcript,
            denoisy_transcript,
            noisy_transcript,
            gt_transcript,
        )
        return loss.mean(), stats

    def eval(self, speech, denoisy_speech, noisy_speech=None, gt_transcript=None):
        reference_transcript = self.inference_without_grad(speech)
        with torch.no_grad():
            denoisy_transcript, logprob = self.inference_with_grad(denoisy_speech)
        reward = [
            -self.asr_metric(preds=[p], target=[r])
            for p, r in zip(denoisy_transcript, reference_transcript)
        ]
        if self.baseline:
            reward = [i + self.baseline for i in reward]
        reward = torch.Tensor(reward).to(self.device)
        loss = -logprob * reward

        if noisy_speech is not None:
            noisy_transcript = self.inference_without_grad(noisy_speech)
        else:
            noisy_transcript = None

        stats = self.get_stats(
            loss,
            reward,
            logprob,
            reference_transcript,
            denoisy_transcript,
            noisy_transcript,
            gt_transcript,
        )
        return stats

    def get_stats(
        self,
        loss,
        reward,
        logprob,
        reference_transcript,
        denoisy_transcript,
        noisy_transcript=None,
        gt_transcript=None,
    ):
        stats = {}
        stats["asr_loss"] = loss.mean().item()
        stats["reward"] = reward.mean().item()
        stats["logprob"] = logprob.mean().item()
        if gt_transcript:
            gt_transcript = [self.normalize_text(i) for i in gt_transcript]
        # metrics between reference and denoisy transcript (reference = clean audio through asr)
        stats["wer (ref-denoisy)"] = self.wer(
            target=reference_transcript, preds=denoisy_transcript
        ).item()
        stats["cer (ref-denoisy)"] = self.cer(
            target=reference_transcript, preds=denoisy_transcript
        ).item()
        # metrics between GT and reference
        if gt_transcript:
            stats["wer (gt-ref)"] = self.wer(
                target=gt_transcript, preds=reference_transcript
            ).item()
            stats["cer (gt-ref)"] = self.cer(
                target=gt_transcript, preds=reference_transcript
            ).item()
        # metrics between GT and deniosy
        if gt_transcript:
            stats["wer (gt-denoisy)"] = self.wer(
                target=gt_transcript, preds=denoisy_transcript
            ).item()
            stats["cer (gt-denoisy)"] = self.cer(
                target=gt_transcript, preds=denoisy_transcript
            ).item()
        # metrics between GT and noisy transcript (without denoising)
        if gt_transcript and noisy_transcript:
            stats["wer (gt-noisy)"] = self.wer(
                target=gt_transcript, preds=noisy_transcript
            ).item()
            stats["cer (gt-noisy)"] = self.cer(
                target=gt_transcript, preds=noisy_transcript
            ).item()

        return stats

    @staticmethod
    def normalize_text(text: str):
        for char in [".", ",", "!", "?", "(", ")"]:
            text = text.replace(char, " ")
        text = text.replace("ё", "е")
        text = re.sub(" +", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower().strip()
        return text

    @staticmethod
    def to_tensor(speech):
        if not torch.is_tensor(speech):
            speech = torch.Tensor(speech)
        if len(speech.shape) == 1:
            speech = speech.unsqueeze(0)
        return speech
