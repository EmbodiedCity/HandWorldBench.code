"""Single-sequence, finite final answer processor; CPU-testable and model-agnostic.

Callable interface is compatible with transformers.LogitsProcessorList without
importing transformers. Construct a fresh instance per generation. Token IDs
must come from the actual tokenizer, including the full legal label set (the
unknown label included). Thinking logits are returned unchanged. This controls
format and budget only; it never retries or prefers a score value.

The caller must allow at least thinking_budget + len(close_ids) +
len(separator_ids) + 2 generated tokens to accommodate forced closure.
"""
import math
import torch


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def _sequence(values, name, unique=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(name + ' must be a nonempty token sequence')
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(name + ' must be a nonempty token sequence') from exc
    if not result:
        raise ValueError(name + ' must be nonempty')
    for value in result:
        _integer(value, name + ' token')
    if unique and len(set(result)) != len(result):
        raise ValueError(name + ' must contain unique token IDs')
    return result


class FiniteAnswer:
    def __init__(self, input_length, thinking_budget, close_ids, separator_ids,
                 label_ids, eos_id):
        self.input_length = _integer(input_length, 'input_length')
        self.thinking_budget = _integer(thinking_budget, 'thinking_budget')
        self.close_ids = _sequence(close_ids, 'close_ids')
        self.separator_ids = _sequence(separator_ids, 'separator_ids')
        self.label_ids = _sequence(label_ids, 'label_ids', unique=True)
        self.eos_id = _integer(eos_id, 'eos_id')
        if self.eos_id in self.close_ids + self.separator_ids + self.label_ids:
            raise ValueError('EOS must not occur in closing, separator or legal labels')
        self.forced_close = False
        self.final_label_probability_mass = None
        self.selected_token = None
        self._forced_start = None
        self._closing_end = None
        self._last_length = input_length
        self._last_prefix = None

    @staticmethod
    def _only(scores, token):
        masked = torch.full_like(scores, -float('inf'))
        masked[0, token] = 0
        return masked

    def _latest_closing(self, generated):
        size = len(self.close_ids)
        for start in range(len(generated) - size, -1, -1):
            if tuple(generated[start:start + size]) == self.close_ids:
                return start + size
        return None

    def __call__(self, input_ids, scores):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('Only one input sequence is supported')
        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError('Only one scores row is supported')
        if not scores.is_floating_point():
            raise ValueError('Scores must be floating point')
        if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('input_ids must contain integer tokens')
        length = input_ids.shape[1]
        if length < self.input_length or length < self._last_length:
            raise ValueError('Input prefix shortened; use a fresh processor per generation')
        vocab = scores.shape[1]
        if any(token >= vocab for token in self.close_ids + self.separator_ids + self.label_ids + (self.eos_id,)):
            raise ValueError('Configured token ID exceeds vocabulary')
        generated = input_ids[0, self.input_length:].tolist()
        if self._last_prefix is not None and generated[:len(self._last_prefix)] != self._last_prefix:
            raise ValueError('Generation prefix changed; only one monotonic sequence is supported')
        self._last_length = length
        self._last_prefix = generated

        # During a forced multi-token close, finish that exact sequence even
        # if a partial natural prefix was present at the budget boundary.
        if self._forced_start is not None and self._closing_end is None:
            consumed = len(generated) - self._forced_start
            expected = self.close_ids[:min(consumed, len(self.close_ids))]
            if tuple(generated[self._forced_start:self._forced_start + len(expected)]) != expected:
                raise ValueError('Forced closing tokens were not followed')
            if consumed < len(self.close_ids):
                return self._only(scores, self.close_ids[consumed])
            self._closing_end = self._forced_start + len(self.close_ids)

        if self._closing_end is None:
            # Ignore closing tokens in the prompt; when an existing generated
            # fragment contains several complete closings, take the latest.
            self._closing_end = self._latest_closing(generated)
            if self._closing_end is None:
                if len(generated) < self.thinking_budget:
                    return scores
                self.forced_close = True
                self._forced_start = len(generated)
                return self._only(scores, self.close_ids[0])

        # Anchor the accepted closing once final-answer formatting starts, so
        # separator/label tokens overlapping structural IDs cannot restart it.
        tail = generated[self._closing_end:]
        separator_length = len(self.separator_ids)
        consumed_separator = min(len(tail), separator_length)
        if tuple(tail[:consumed_separator]) != self.separator_ids[:consumed_separator]:
            raise ValueError('Generated separator differs from configured sequence')
        if len(tail) < separator_length:
            return self._only(scores, self.separator_ids[len(tail)])
        if len(tail) == separator_length:
            legal = scores[0, list(self.label_ids)]
            if torch.isnan(legal).any() or torch.isneginf(legal).all():
                raise ValueError('Legal label logits have no defined argmax')
            selected = self.label_ids[int(torch.argmax(legal).item())]
            self.selected_token = selected
            # Diagnostic only; never used to select or retry an answer.
            with torch.no_grad():
                mass = torch.exp(torch.logsumexp(legal.double(), dim=0) -
                                 torch.logsumexp(scores[0].double(), dim=0)).item()
            self.final_label_probability_mass = min(1., max(0., mass)) if math.isfinite(mass) else None
            return self._only(scores, selected)
        actual_label = tail[separator_length]
        if actual_label not in self.label_ids:
            raise ValueError('Generated final token is not a legal label')
        if self.selected_token is not None and actual_label != self.selected_token:
            raise ValueError('Selected final label was not followed')
        self.selected_token = actual_label
        if any(token != self.eos_id for token in tail[separator_length + 1:]):
            raise ValueError('Unexpected tokens after final label')
        return self._only(scores, self.eos_id)
