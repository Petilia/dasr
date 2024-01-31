import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from ..metrics import TruncatedCER, TruncatedWER


class BaseASRModel:
    """Base class for ASR models, that returns asr loss and stats.
    Methods get_loss, eval, inference_with_grad and transcribe must be implemented in child classes.
    """

    def __init__(self) -> None:
        self.wer = TruncatedWER()
        self.cer = TruncatedCER()

    def get_loss(
        self,
        speech: Tensor,
        denoisy_speech: Tensor,
        attention_mask: Tensor = None,
        noisy_speech: List[str] = None,
        gt_transcript: List[str] = None,
    ) -> Tuple[Dict[str, float], Tensor]:
        "Method, that returns stats and loss for the model."
        raise NotImplementedError("Method must be implemented")

    def eval(self) -> Dict[str, float]:
        "Method, that returns stats for the model on inference"
        raise NotImplementedError("Method must be implemented")

    def inference_with_grad(
        self, speech: Tensor, attention_mask: Tensor = None
    ) -> Tuple[List[str], Tensor]:
        "Method, that returns predicted text and logprobs for the model. Used for REINFORCE"
        raise NotImplementedError("Method must be implemented")

    def transcribe(self, speech: Tensor, attention_mask: Tensor = None) -> List[str]:
        "Method, that returns predicted text"
        raise NotImplementedError("Method must be implemented")

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_stats(
        self,
        loss: Tensor,
        reward: Tensor,
        logprob: Tensor,
        reference_transcript: List[str],
        denoisy_transcript: List[str],
        noisy_transcript: List[str] = None,
        gt_transcript: List[str] = None,
    ):
        stats = {}
        stats["asr_loss"] = loss.item()
        if reward is not None:
            stats["reward"] = reward.mean().item()
        if logprob is not None:
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
