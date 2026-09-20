import evaluate
import numpy as np

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def get_compute_metrics_fn(tokenizer):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 with pad_token_id so we can decode
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Truncate each prediction at the first </s>: everything from the predicted
        # end-of-sequence onward (the <fill> tail) is dropped so a stray post-eos
        # character cannot leak into the hypothesis and inflate WER.
        eos_id = tokenizer.eos_token_id
        for i in range(pred_ids.shape[0]):
            hits = np.where(pred_ids[i] == eos_id)[0]
            if hits.size:
                pred_ids[i, hits[0]:] = tokenizer.pad_token_id

        # Since we use a Wav2Vec2CTCTokenizer but without CTC,
        # we must decode without group_tokens=True so it doesn't merge adjacent identical letters.

        # Extract text by just using normal decoding
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)

        # Compute metrics
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}
    return compute_metrics
