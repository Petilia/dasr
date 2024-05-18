import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .base import BaseASRModel
from ..metrics import TruncatedCER, TruncatedWER


class Wav2VecEnv(torch.nn.Module):
    def __init__(
        self,
        path_model
    ):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(path_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(path_model)

        self.model.eval()
        self.freeze_model()
        self.model.freeze_feature_extractor()

        self.wer = TruncatedWER()
        self.cer = TruncatedCER()

    def transcribe(self, speech: Tensor, attention_mask: Tensor = None) -> List[str]:
        "Method, that returns predicted text"
        speech = (speech - speech.mean()) / (speech.std() + 1e-7)  # normalize
        with torch.no_grad():
            output = self.model(speech, attention_mask=attention_mask)

        tokens_logits = output.logits
        predicted_ids = torch.argmax(tokens_logits, dim=-1)
        pred_texts = self.processor.batch_decode(predicted_ids)

        return pred_texts

    def forward(
        self,
        denoisy_speech: Tensor,
        attention_mask: Tensor = None,
        gt_transcript: List[str] = None,
    ):
        "Forward method"
        target = [self.normalize_text(t) for t in gt_transcript]
        target_ids = self.processor(
            text=target, padding=True, return_tensors="pt"
        ).input_ids

        # replace padding tokens with -100 to ignore them for loss calculation
        target_ids[target_ids == self.processor.tokenizer.pad_token_id] = -100

        output = self.model(
            denoisy_speech,
            attention_mask=attention_mask,
            labels=target_ids,
        )

        return output

    def get_loss(
        self,
        output,
        speech: Tensor,
        attention_mask: Tensor = None,
        noisy_speech: List[str] = None,
        gt_transcript: List[str] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        "Method, that returns stats and loss for the model."

        pred_ids = torch.argmax(output.logits, dim=-1)
        denoisy_transcript = self.processor.batch_decode(pred_ids)

        loss = output.loss

        if noisy_speech is not None:
            noisy_transcript = self.transcribe(noisy_speech)
        else:
            noisy_transcript = None

        reference_transcript = self.transcribe(speech, attention_mask=attention_mask)

        stats = self.get_stats(
            loss,
            reference_transcript,
            denoisy_transcript,
            noisy_transcript,
            gt_transcript,
        )

        return loss, stats
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_stats(
        self,
        loss: Tensor,
        reference_transcript: List[str],
        denoisy_transcript: List[str],
        noisy_transcript: List[str] = None,
        gt_transcript: List[str] = None,
    ):
        stats = {}
        stats["asr_loss"] = loss.item()
        if gt_transcript:
            gt_transcript = [self.normalize_text(i) for i in gt_transcript]
        # metrics between GT and reference (reference = clean audio through asr)
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





# class Wav2VecEnv(BaseASRModel, torch.nn.Module):
#     def __init__(
#         self,
#         path_model
#     ):
#         super().__init__()
#         self.processor = Wav2Vec2Processor.from_pretrained(path_model)
#         self.model = Wav2Vec2ForCTC.from_pretrained(path_model)

#         self.model.eval()
#         self.freeze_model()

#     def transcribe(self, speech: Tensor, attention_mask: Tensor = None) -> List[str]:
#         "Method, that returns predicted text"
#         speech = (speech - speech.mean()) / (speech.std() + 1e-7)  # normalize
#         with torch.no_grad():
#             output = self.model(speech, attention_mask=attention_mask)

#         tokens_logits = output.logits
#         predicted_ids = torch.argmax(tokens_logits, dim=-1)
#         pred_texts = self.processor.batch_decode(predicted_ids)

#         return pred_texts

#     def forward(
#         self,
#         denoisy_speech: Tensor,
#         attention_mask: Tensor = None,
#         gt_transcript: List[str] = None,
#     ):
#         "Forward method"
#         target = [self.normalize_text(t) for t in gt_transcript]
#         target_ids = self.processor(
#             text=target, padding=True, return_tensors="pt"
#         ).input_ids

#         # replace padding tokens with -100 to ignore them for loss calculation
#         target_ids[target_ids == self.processor.tokenizer.pad_token_id] = -100

#         output = self.model(
#             denoisy_speech,
#             attention_mask=attention_mask,
#             labels=target_ids,
#         )

#         return output

#     def get_loss(
#         self,
#         output,
#         speech: Tensor,
#         attention_mask: Tensor = None,
#         noisy_speech: List[str] = None,
#         gt_transcript: List[str] = None,
#     ) -> Tuple[Tensor, Dict[str, float]]:
#         "Method, that returns stats and loss for the model."

#         pred_ids = torch.argmax(output.logits, dim=-1)
#         denoisy_transcript = self.processor.batch_decode(pred_ids)

#         loss = output.loss

#         if noisy_speech is not None:
#             noisy_transcript = self.transcribe(noisy_speech)
#         else:
#             noisy_transcript = None

#         reference_transcript = self.transcribe(speech, attention_mask=attention_mask)

#         stats = self.get_stats(
#             loss,
#             reference_transcript,
#             denoisy_transcript,
#             noisy_transcript,
#             gt_transcript,
#         )

#         return loss, stats
