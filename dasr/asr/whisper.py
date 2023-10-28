import re
from types import MethodType

import torch
import torch.nn.functional as F
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from undecorated import undecorated

from .whisper_utils import log_mel_spectrogram, pad_or_trim


class WhisperEnv:
    def __init__(
        self,
        device="cuda",
        path_model="/home/docker_current/hf_whisper/whisper-base",
        asr_metric="wer",
        baseline=False,
    ):
        self.device = device
        self.model = WhisperForConditionalGeneration.from_pretrained(path_model)
        self.processor = WhisperProcessor.from_pretrained(
            path_model, language="ru", task="transcribe"
        )
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="ru", task="transcribe"
        )
        # add method for generate with gradients
        generate_with_grad = undecorated(self.model.generate)
        self.model.generate_with_grad = MethodType(generate_with_grad, self.model)
        self.model = self.model.to(device)
        self.model.eval()
        self.freeze_model()

        self.wer = load("wer")
        self.cer = load("cer")
        if asr_metric == "wer":
            self.asr_metric = self.wer
        elif asr_metric == "cer":
            self.asr_metric = self.cer
        self.baseline = baseline

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_melspec(self, speech):
        speech = self.to_tensor(speech)
        speech = speech.to(self.device)
        speech = pad_or_trim(speech)
        orig_whisper_feat = log_mel_spectrogram(speech)
        return orig_whisper_feat

    def inference_with_grad(self, speech):
        mel_features = self.get_melspec(speech)
        output = self.model.generate_with_grad(
            mel_features,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=50,
        )
        predicted_ids = output[0]
        pred_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        pred_text = [self.normalize_text(i) for i in pred_text]
        tokens_logits = output[1]
        logprob = sum(
            [F.softmax(logits, dim=1).max(dim=1)[0].log() for logits in tokens_logits]
        )
        return pred_text, logprob

    def inference_without_grad(self, speech):
        mel_features = self.get_melspec(speech)
        output = self.model.generate(mel_features, max_length=50)
        pred_text = self.processor.batch_decode(output, skip_special_tokens=True)
        pred_text = [self.normalize_text(i) for i in pred_text]
        return pred_text

    def get_loss(self, speech, denoisy_speech, noisy_speech=None, gt_transcript=None):
        reference_transcript = self.inference_without_grad(speech)
        denoisy_transcript, logprob = self.inference_with_grad(denoisy_speech)
        reward = [
            -self.asr_metric.compute(predictions=[p], references=[r])
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
            -self.asr_metric.compute(predictions=[p], references=[r])
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
        stats["wer (ref-denoisy)"] = self.wer.compute(
            references=reference_transcript, predictions=denoisy_transcript
        )
        stats["cer (ref-denoisy)"] = self.cer.compute(
            references=reference_transcript, predictions=denoisy_transcript
        )
        # metrics between GT and reference
        if gt_transcript:
            stats["wer (gt-ref)"] = self.wer.compute(
                references=gt_transcript, predictions=reference_transcript
            )
            stats["cer (gt-ref)"] = self.cer.compute(
                references=gt_transcript, predictions=reference_transcript
            )
        # metrics between GT and deniosy
        if gt_transcript:
            stats["wer (gt-denoisy)"] = self.wer.compute(
                references=gt_transcript, predictions=denoisy_transcript
            )
            stats["cer (gt-denoisy)"] = self.cer.compute(
                references=gt_transcript, predictions=denoisy_transcript
            )
        # metrics between GT and noisy transcript (without denoising)
        if gt_transcript and noisy_transcript:
            stats["wer (gt-noisy)"] = self.wer.compute(
                references=gt_transcript, predictions=noisy_transcript
            )
            stats["cer (gt-noisy)"] = self.cer.compute(
                references=gt_transcript, predictions=noisy_transcript
            )

        # print("GT")
        # print(gt_transcript)
        # print("reference")
        # print(reference_transcript)
        # print("noisy")
        # print(noisy_transcript)
        # print("denoisy")
        # print(denoisy_transcript)
        # # metrics between reference and noisy (какая-то странная метрика между asr на чистом и зашумленном звуке)
        # if noisy_transcript:
        #     stats["wer (ref-noisy)"] = self.wer.compute(references=reference_transcript,
        #                                     predictions=noisy_transcript)
        #     stats["cer (ref-noisy)"] = self.cer.compute(references=reference_transcript,
        #                                     predictions=noisy_transcript)
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
