from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import random

from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec, MemoryDataset

@dataclass
class MCQSample:
    id: str
    question: str
    options: List[str]
    correct_index: Optional[int]  # if known, else None

# -------------------------
# Helpers
# -------------------------

def _stable_seed_from_text(text: str) -> int:
    h = hashlib.sha256(text.encode()).hexdigest()
    return int(h[:8], 16)

def _stable_rng(seed_text: str) -> random.Random:
    return random.Random(_stable_seed_from_text(seed_text))

def _ensure_str_list(xs):
    return [str(x) for x in xs]

# -------------------------
# Existing loaders (kept)
# -------------------------

def _mmlu(subject: str) -> Dataset:
    return hf_dataset(
        "cais/mmlu",
        split="test",
        name=subject,
        sample_fields=FieldSpec(input="question", target="answer", metadata=["choices"])
    )

def mmlu_mcq(subject="high_school_physics") -> Dataset:
    ds = _mmlu(subject)
    samples = []
    for sample in ds:
        choices = _ensure_str_list(sample.metadata["choices"])
        # target is 'A','B','C','D' or 0..3; normalize
        t = sample.target
        if isinstance(t, str):
            correct_idx = ord(t.upper()) - ord('A') if t.isalpha() else int(t)
        else:
            correct_idx = int(t)
        s = Sample(
            id=getattr(sample, "id", None) or f"mmlu_{subject}_{len(samples)}",
            input=sample.input,
            target=str(correct_idx),
            metadata={"options": choices}
        )
        samples.append(s)
    return MemoryDataset(samples)

def arc_challenge_mcq() -> Dataset:
    ds = hf_dataset(
        "allenai/ai2_arc",
        split="test",
        name="ARC-Challenge",
        sample_fields=FieldSpec(input="question", target="answerKey", metadata=["choices"])
    )
    samples = []
    for i, sample in enumerate(ds):
        choices = sample.metadata["choices"]
        labels = choices.get("label", [])
        texts = _ensure_str_list(choices.get("text", []))
        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        correct_label = sample.target.strip()
        correct_idx = label_to_idx.get(correct_label, 0)
        s = Sample(
            id=getattr(sample, "id", None) or f"arc_challenge_{i}",
            input=sample.input,
            target=str(correct_idx),
            metadata={"options": texts}
        )
        samples.append(s)
    return MemoryDataset(samples)

def ethics_pair_to_mcq(name="commonsense") -> Dataset:
    ds = hf_dataset("wassname/ethics_expression_preferences",
                    split="train",
                    name=name,
                    sample_fields=FieldSpec(id="task_id", input="prompt", target="chosen", metadata=["rejected"]))
    samples = []
    for i, sample in enumerate(ds):
        chosen = str(sample.target)
        rejected = str(sample.metadata.get("rejected", ""))
        options = [chosen, rejected, chosen + " (variant A)", rejected + " (variant B)"]
        s = Sample(
            id=getattr(sample, "id", None) or f"ethics_{name}_{i}",
            input=sample.input,
            target="0",
            metadata={"options": options}
        )
        samples.append(s)
    return MemoryDataset(samples)

# -------------------------
# New loaders (shortlist)
# -------------------------

def arc_easy_mcq() -> Dataset:
    ds = hf_dataset(
        "allenai/ai2_arc",
        split="test",
        name="ARC-Easy",
        sample_fields=FieldSpec(input="question", target="answerKey", metadata=["choices"])
    )
    samples = []
    for i, sample in enumerate(ds):
        choices = sample.metadata["choices"]
        labels = choices.get("label", [])
        texts = _ensure_str_list(choices.get("text", []))
        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        correct_label = sample.target.strip()
        correct_idx = label_to_idx.get(correct_label, 0)
        s = Sample(
            id=getattr(sample, "id", None) or f"arc_easy_{i}",
            input=sample.input,
            target=str(correct_idx),
            metadata={"options": texts}
        )
        samples.append(s)
    return MemoryDataset(samples)

def hellaswag_mcq(split: str = "validation") -> Dataset:
    """
    HellaSwag: 4 endings, label in [0..3]
    Fields: ctx (context/prefix), endings (List[str]), label (int)
    """
    ds = hf_dataset(
        "hellaswag",
        split=split,
        sample_fields=FieldSpec(input="ctx", target="label", metadata=["endings"])
    )
    samples = []
    for i, sample in enumerate(ds):
        options = _ensure_str_list(sample.metadata["endings"])
        correct_idx = int(sample.target)
        # Build a readable question
        question = f"{sample.input}\n\nChoose the most plausible ending."
        s = Sample(
            id=getattr(sample, "id", None) or f"hellaswag_{split}_{i}",
            input=question,
            target=str(correct_idx),
            metadata={"options": options}
        )
        samples.append(s)
    return MemoryDataset(samples)

def commonsenseqa_mcq(split: str = "validation") -> Dataset:
    """
    CommonsenseQA has 5 options per question. We deterministically pick the correct + 3 distractors.
    Fields: question (str), choices: {'label': [A..E], 'text': [str x5]}, answerKey: 'A'..'E'
    """
    ds = hf_dataset(
        "commonsense_qa",
        split=split,
        sample_fields=FieldSpec(input="question", target="answerKey", metadata=["choices"])
    )
    samples = []
    for i, sample in enumerate(ds):
        choices = sample.metadata["choices"]
        labels = choices.get("label", [])
        texts = _ensure_str_list(choices.get("text", []))
        # Map correct
        correct_label = sample.target.strip()
        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        full_correct_idx = label_to_idx.get(correct_label, 0)

        # Choose 3 distractors deterministically
        distractor_indices = [idx for idx in range(len(texts)) if idx != full_correct_idx]
        rng = _stable_rng(getattr(sample, "id", None) or f"{sample.input}-{i}")
        rng.shuffle(distractor_indices)
        chosen_distractors = distractor_indices[:3]

        # Assemble 4 options: correct + 3 chosen distractors; then shuffle deterministically for balance
        four_options = [texts[full_correct_idx]] + [texts[j] for j in chosen_distractors]
        # Stable shuffle with a different seed to avoid trivial position bias
        rng2 = _stable_rng(f"shuffle-{getattr(sample, 'id', None) or f'{sample.input}-{i}'}")
        rng2.shuffle(four_options)

        # Find new correct index
        correct_idx = four_options.index(texts[full_correct_idx])

        s = Sample(
            id=getattr(sample, "id", None) or f"csqa_{split}_{i}",
            input=sample.input,
            target=str(correct_idx),
            metadata={"options": four_options}
        )
        samples.append(s)
    return MemoryDataset(samples)

def sciq_mcq(split: str = "validation") -> Dataset:
    """
    SciQ: one correct + 3 distractors provided.
    Fields: question (str), correct_answer (str), distractor1/2/3 (str)
    """
    ds = hf_dataset(
        "sciq",
        split=split,
        sample_fields=FieldSpec(input="question", target="correct_answer", metadata=["distractor1", "distractor2", "distractor3"])
    )
    samples = []
    for i, sample in enumerate(ds):
        correct = str(sample.target)
        d1 = str(sample.metadata.get("distractor1", ""))
        d2 = str(sample.metadata.get("distractor2", ""))
        d3 = str(sample.metadata.get("distractor3", ""))
        options = [correct, d1, d2, d3]
        # Stable shuffle so correct isn’t always at position 0
        rng = _stable_rng(getattr(sample, "id", None) or f"sciq-{i}-{sample.input}")
        rng.shuffle(options)
        correct_idx = options.index(correct)
        s = Sample(
            id=getattr(sample, "id", None) or f"sciq_{split}_{i}",
            input=sample.input,
            target=str(correct_idx),
            metadata={"options": options}
        )
        samples.append(s)
    return MemoryDataset(samples)

def openbookqa_mcq(split: str = "validation") -> Dataset:
    """
    OpenBookQA main config.
    Fields: question_stem (str), choices: {'label': ['A'..'D'], 'text': [str x4]}, answerKey: 'A'..'D'
    """
    ds = hf_dataset(
        "openbookqa",
        split=split,
        name="main",
        sample_fields=FieldSpec(input="question_stem", target="answerKey", metadata=["choices"])
    )
    samples = []
    for i, sample in enumerate(ds):
        choices = sample.metadata["choices"]
        labels = choices.get("label", [])
        texts = _ensure_str_list(choices.get("text", []))
        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        correct_label = sample.target.strip()
        correct_idx = label_to_idx.get(correct_label, 0)
        s = Sample(
            id=getattr(sample, "id", None) or f"openbookqa_{split}_{i}",
            input=sample.input,
            target=str(correct_idx),
            metadata={"options": texts}
        )
        samples.append(s)
    return MemoryDataset(samples)

def medmcqa_mcq(split: str = "validation") -> Dataset:
    """
    MedMCQA (robust to common field layouts).
    Attempts to support both:
      - 'options': [str x4], 'cop'/'answer_idx'/'answer' for label
      - or individual fields: 'opa','opb','opc','opd' + 'cop' (letter)
    """
    ds = hf_dataset(
        "medmcqa",
        split=split,
        sample_fields=FieldSpec(
            input="question",
            target="cop",  # common: correct option letter; we’ll normalize below
            metadata=["options", "opa", "opb", "opc", "opd", "answer_idx", "answer"]
        )
    )
    samples = []
    for i, sample in enumerate(ds):
        # Build options
        options = sample.metadata.get("options")
        if not options:
            # Fallback to opa..opd
            ops = [sample.metadata.get("opa"), sample.metadata.get("opb"),
                   sample.metadata.get("opc"), sample.metadata.get("opd")]
            options = [str(x) for x in ops if x is not None]
        options = _ensure_str_list(options)
        if len(options) < 4:
            # Skip malformed items
            continue

        # Determine correct index
        correct_idx = None
        t = sample.target
        # Common cases
        if isinstance(t, str):
            ts = t.strip()
            if ts.upper() in ["A", "B", "C", "D"]:
                correct_idx = ord(ts.upper()) - ord('A')
            else:
                # Maybe t is the correct answer text; map to index
                if ts in options:
                    correct_idx = options.index(ts)
        elif isinstance(t, int):
            correct_idx = int(t)

        # Other hints in metadata
        if correct_idx is None:
            if "answer_idx" in sample.metadata and sample.metadata["answer_idx"] is not None:
                try:
                    correct_idx = int(sample.metadata["answer_idx"])
                except Exception:
                    pass
        if correct_idx is None and "answer" in sample.metadata and sample.metadata["answer"] is not None:
            ans = str(sample.metadata["answer"])
            if ans.upper() in ["A", "B", "C", "D"]:
                correct_idx = ord(ans.upper()) - ord('A')
            elif ans in options:
                correct_idx = options.index(ans)

        if correct_idx is None:
            # As a last resort, default to 0
            correct_idx = 0

        s = Sample(
            id=getattr(sample, "id", None) or f"medmcqa_{split}_{i}",
            input=str(sample.input),
            target=str(correct_idx),
            metadata={"options": options[:4]}
        )
        samples.append(s)
    return MemoryDataset(samples)


# -------------------------
# Dispatcher
# -------------------------

def load_dataset(name: str, **kwargs) -> Dataset:
    """
    Names:
      - mmlu:<subject> (e.g., mmlu:high_school_physics)
      - arc_challenge
      - arc_easy
      - hellaswag[:split] (default split=validation)
      - commonsenseqa[:split] (default split=validation)
      - sciq[:split] (default split=validation)
      - openbookqa[:split] (default split=validation)
      - medmcqa[:split] (default split=validation)
      - ethics:<subset> (commonsense, deontology, justice, utilitarianism, virtue, ...)
    """
    if name.startswith("mmlu:"):
        return mmlu_mcq(subject=name.split(":")[1])

    if name == "arc_challenge":
        return arc_challenge_mcq()
    if name == "arc_easy":
        return arc_easy_mcq()

    if name.startswith("hellaswag"):
        parts = name.split(":")
        split = parts[1] if len(parts) > 1 else "validation"
        return hellaswag_mcq(split=split)

    if name.startswith("commonsenseqa"):
        parts = name.split(":")
        split = parts[1] if len(parts) > 1 else "validation"
        return commonsenseqa_mcq(split=split)

    if name.startswith("sciq"):
        parts = name.split(":")
        split = parts[1] if len(parts) > 1 else "validation"
        return sciq_mcq(split=split)

    if name.startswith("openbookqa"):
        parts = name.split(":")
        split = parts[1] if len(parts) > 1 else "validation"
        return openbookqa_mcq(split=split)

    if name.startswith("medmcqa"):
        parts = name.split(":")
        split = parts[1] if len(parts) > 1 else "validation"
        return medmcqa_mcq(split=split)

    if name.startswith("ethics:"):
        return ethics_pair_to_mcq(name=name.split(":")[1])

    raise ValueError(f"Unknown dataset name: {name}")