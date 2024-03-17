import re
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .base import BaseASRModel


class Wav2VecEnv(BaseASRModel):
    def __init__(
        self,
        device="cuda",
        path_model="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        loss_type="ctc",
        asr_metric="wer",
        baseline=False,
    ):
        super().__init__()

        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(path_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(path_model)

        self.model = self.model.to(device)
        self.model.eval()
        self.freeze_model()

        self.loss_type = loss_type

        if self.loss_type == "reinforce":
            if asr_metric == "wer":
                self.asr_metric = self.wer
            elif asr_metric == "cer":
                self.asr_metric = self.cer
            self.baseline = baseline

    def inference_with_grad(
        self, speech: Tensor, attention_mask: Tensor = None
    ) -> Tuple[List[str], Tensor]:
        "Method, that returns predicted text and logprobs for the model. Used for REINFORCE"
        speech = (speech - speech.mean()) / (speech.std() + 1e-7)  # normalize

        output = self.model(speech, attention_mask=attention_mask)
        tokens_logits = output.logits
        predicted_ids = torch.argmax(tokens_logits, dim=-1)
        pred_texts = self.processor.batch_decode(predicted_ids)
        logprobs = tokens_logits.softmax(dim=2).max(dim=2).values.log().sum(dim=1)
        return pred_texts, logprobs

    def transcribe(self, speech: Tensor, attention_mask: Tensor = None) -> List[str]:
        "Method, that returns predicted text"
        speech = (speech - speech.mean()) / (speech.std() + 1e-7)  # normalize
        with torch.no_grad():
            output = self.model(speech, attention_mask=attention_mask)

        tokens_logits = output.logits
        predicted_ids = torch.argmax(tokens_logits, dim=-1)
        pred_texts = self.processor.batch_decode(predicted_ids)

        return pred_texts

    def get_loss(
        self,
        speech: Tensor,
        denoisy_speech: Tensor,
        attention_mask: Tensor = None,
        noisy_speech: List[str] = None,
        gt_transcript: List[str] = None,
    ) -> Tuple[Dict[str, float], Tensor]:
        "Method, that returns stats and loss for the model."

        reward = None
        logprob = None
        reference_transcript = self.transcribe(speech, attention_mask=attention_mask)

        if self.loss_type == "reinforce":
            denoisy_transcript, logprob = self.inference_with_grad(
                denoisy_speech, attention_mask=attention_mask
            )
            reward = [
                -self.asr_metric(preds=[p], target=[r])
                for p, r in zip(denoisy_transcript, reference_transcript)
            ]
            if self.baseline:
                reward = [i + self.baseline for i in reward]
            reward = torch.Tensor(reward).to(self.device)
            loss = -logprob * reward
            loss = loss.mean()

        if self.loss_type == "ctc":
            target = [self.normalize_text(t) for t in gt_transcript]
            target_ids = self.processor(
                text=target, padding=True, return_tensors="pt"
            ).input_ids

            # replace padding tokens with -100 to ignore them for loss calculation
            target_ids[target_ids == self.processor.tokenizer.pad_token_id] = -100

            output = self.model(
                denoisy_speech,
                attention_mask=attention_mask,
                labels=target_ids.to(self.device),
            )

            pred_ids = torch.argmax(output.logits, dim=-1)
            denoisy_transcript = self.processor.batch_decode(pred_ids)

            loss = output.loss

        if noisy_speech is not None:
            noisy_transcript = self.transcribe(noisy_speech)
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

        return loss, stats

    def eval(
        self,
        speech: Tensor,
        denoisy_speech: Tensor,
        attention_mask: Tensor = None,
        noisy_speech: List[str] = None,
        gt_transcript: List[str] = None,
    ) -> Dict[str, float]:
        "Method, that returns stats for the model on inference"
        with torch.no_grad():
            _, stats = self.get_loss(
                speech,
                denoisy_speech,
                attention_mask=attention_mask,
                noisy_speech=noisy_speech,
                gt_transcript=gt_transcript,
            )

        return stats
