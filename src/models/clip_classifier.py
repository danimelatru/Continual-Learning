from transformers import AutoModelForImageClassification

def load_clip_classifier(checkpoint, num_labels=10):
    return AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )