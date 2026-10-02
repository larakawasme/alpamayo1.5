"""Metrics from THUDM/LongBench (v1) metrics.py and eval.py.

Upstream's code_sim_score uses fuzzywuzzy, whose requirements do not include
python-Levenshtein, so fuzz.ratio falls back to difflib; `_fuzz_ratio` is that
fallback reimplemented with the standard library.
"""

import re
import string
from collections import Counter
from difflib import SequenceMatcher

import jieba
from rouge import Rouge


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    s = "".join(ch for ch in s.lower() if ch not in all_punctuation)
    return "".join(s.split())


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return 0.0 if not numbers else right_num / len(numbers)


def _retrieval(prediction, ground_truth, pattern):
    ground_truth_id = re.findall(pattern, ground_truth)[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return 0.0 if not numbers else right_num / len(numbers)


def retrieval_score(prediction, ground_truth, **kwargs):
    return _retrieval(prediction, ground_truth, r"Paragraph (\d+)")


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    return _retrieval(prediction, ground_truth, r"段落(\d+)")


def _fuzz_ratio(s1, s2):
    """fuzzywuzzy.fuzz.ratio with the difflib backend: 0-100 integer."""
    if s1 == s2:
        return 100
    if not s1 or not s2:
        return 0
    return int(round(100 * SequenceMatcher(None, s1, s2).ratio()))


def first_code_line(prediction):
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return line
    return ""


def code_sim_score(prediction, ground_truth, **kwargs):
    return _fuzz_ratio(first_code_line(prediction), ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = [c for c in kwargs["all_classes"] if c in prediction]
    # Removing while iterating skips elements; kept as upstream for identical scores.
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def rouge_score(prediction, ground_truth, **kwargs):
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(jieba.cut(prediction, cut_all=False))
    ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
    return rouge_score(prediction, ground_truth)


def f1_score(prediction, ground_truth, **kwargs):
    num_same = sum((Counter(prediction) & Counter(ground_truth)).values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    def tokens(text):
        normalized = (normalize_zh_answer(t) for t in jieba.cut(text, cut_all=False))
        return [t for t in normalized if t]

    return f1_score(tokens(prediction), tokens(ground_truth))


DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


def scored_prediction(dataset, prediction):
    """The text the metric sees (few-shot tasks are scored on the first line only)."""
    if dataset in FIRST_LINE_TASKS:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def sample_score(dataset, prediction, answers, all_classes):
    prediction = scored_prediction(dataset, prediction)
    metric = DATASET2METRIC[dataset]
    score = 0.0
    for ground_truth in answers:
        score = max(score, metric(prediction, ground_truth, all_classes=all_classes))
    return score
