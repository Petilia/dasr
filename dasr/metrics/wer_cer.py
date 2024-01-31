from torchmetrics.text import CharErrorRate, WordErrorRate


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
