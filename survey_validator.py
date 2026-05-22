"""
Survey Human Validation Script

Detects bot/automated responses using multiple heuristics:
  - Completion time analysis
  - Straight-lining (same answer repeated across scale questions)
  - Speeder detection (completed too fast)
  - Flatline detection (all identical Likert responses)
  - Open-ended response quality (length, gibberish, copy-paste)
  - Honeypot field detection
  - Consistency/logic checks
  - Overall suspicion scoring
"""

import re
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SurveyResponse:
    """
    Represents a single survey submission.

    Fields
    ------
    respondent_id     : unique identifier for the respondent
    completion_time_s : seconds taken to complete the survey (None = unknown)
    scale_responses   : list of numeric Likert/scale answers (e.g. 1-5)
    open_ended        : dict of question_key -> free-text answer
    honeypot_fields   : dict of hidden field name -> submitted value
                        (legitimate humans should leave these blank/None)
    metadata          : optional dict for IP, user-agent, etc.
    """
    respondent_id: str
    completion_time_s: Optional[float]
    scale_responses: list[int]
    open_ended: dict[str, str] = field(default_factory=dict)
    honeypot_fields: dict[str, Optional[str]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    respondent_id: str
    is_human: bool
    suspicion_score: float          # 0.0 (clean) – 1.0 (definitely bot)
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        verdict = "HUMAN" if self.is_human else "BOT/SUSPICIOUS"
        flags_str = ", ".join(self.flags) if self.flags else "none"
        return (
            f"[{verdict}] {self.respondent_id} "
            f"| score={self.suspicion_score:.2f} "
            f"| flags: {flags_str}"
        )


# ---------------------------------------------------------------------------
# Individual check functions  (each returns a penalty 0.0–1.0 and a flag str)
# ---------------------------------------------------------------------------

def check_honeypot(response: SurveyResponse) -> tuple[float, list[str]]:
    """Bots often fill every field, including hidden honeypots."""
    triggered = [
        k for k, v in response.honeypot_fields.items()
        if v not in (None, "", " ")
    ]
    if triggered:
        return 1.0, [f"honeypot_triggered({', '.join(triggered)})"]
    return 0.0, []


def check_completion_time(
    response: SurveyResponse,
    min_human_seconds: float = 30.0,
    max_human_seconds: float = 3600.0,
) -> tuple[float, list[str]]:
    """
    Responses completed under ~30 s are almost certainly automated.
    Responses over 1 hour may indicate tab-leaving or script pausing,
    but are weighted less harshly.
    """
    if response.completion_time_s is None:
        return 0.0, []

    t = response.completion_time_s
    if t < min_human_seconds:
        penalty = min(1.0, 1.0 - t / min_human_seconds)
        return round(penalty, 3), [f"too_fast({t:.1f}s)"]
    if t > max_human_seconds:
        return 0.2, [f"unusually_slow({t:.1f}s)"]
    return 0.0, []


def check_straightlining(
    response: SurveyResponse,
    threshold: float = 0.9,
) -> tuple[float, list[str]]:
    """
    Straight-lining: respondent picks the same scale value for almost every
    question.  threshold=0.9 means 90 %+ identical answers triggers the flag.
    """
    if len(response.scale_responses) < 4:
        return 0.0, []

    counts = {}
    for v in response.scale_responses:
        counts[v] = counts.get(v, 0) + 1

    most_common_count = max(counts.values())
    ratio = most_common_count / len(response.scale_responses)

    if ratio >= threshold:
        dominant = max(counts, key=counts.__getitem__)
        penalty = (ratio - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
        return round(min(penalty, 1.0), 3), [
            f"straightlining(value={dominant}, ratio={ratio:.0%})"
        ]
    return 0.0, []


def check_scale_variance(
    response: SurveyResponse,
    min_variance: float = 0.2,
) -> tuple[float, list[str]]:
    """
    Very low variance across scale items is a bot signal even when values
    aren't all identical (e.g. alternating 3-4-3-4).
    """
    if len(response.scale_responses) < 4:
        return 0.0, []

    var = statistics.pvariance(response.scale_responses)
    if var < min_variance:
        penalty = 1.0 - (var / min_variance)
        return round(penalty, 3), [f"low_scale_variance({var:.3f})"]
    return 0.0, []


def _is_gibberish(text: str) -> bool:
    """
    Simple gibberish heuristic:
    - Very low ratio of vowels to consonants
    - Long runs of repeated characters
    - High ratio of non-alphabetic characters
    """
    if not text:
        return False
    letters = [c for c in text.lower() if c.isalpha()]
    if len(letters) < 5:
        return True  # too short to be meaningful

    vowels = sum(1 for c in letters if c in "aeiou")
    vowel_ratio = vowels / len(letters)

    non_alpha_ratio = 1.0 - len(letters) / max(len(text), 1)
    has_long_run = bool(re.search(r"(.)\1{4,}", text))  # 5+ repeated chars

    return vowel_ratio < 0.1 or non_alpha_ratio > 0.7 or has_long_run


def check_open_ended(
    response: SurveyResponse,
    min_length: int = 10,
) -> tuple[float, list[str]]:
    """
    Validates free-text responses:
    - Blank or whitespace-only answers
    - Too short (< min_length chars)
    - Gibberish / random character sequences
    - Copy-paste duplicate answers across questions
    """
    if not response.open_ended:
        return 0.0, []

    flags = []
    penalties = []
    seen_answers: dict[str, str] = {}

    for key, text in response.open_ended.items():
        stripped = text.strip() if text else ""

        if not stripped:
            flags.append(f"blank_open_ended({key})")
            penalties.append(0.5)
            continue

        if len(stripped) < min_length:
            flags.append(f"short_open_ended({key}, len={len(stripped)})")
            penalties.append(0.3)

        if _is_gibberish(stripped):
            flags.append(f"gibberish_open_ended({key})")
            penalties.append(0.7)

        # Duplicate answer detection
        normalized = re.sub(r"\s+", " ", stripped.lower())
        if normalized in seen_answers:
            flags.append(
                f"copy_paste_open_ended({key} == {seen_answers[normalized]})"
            )
            penalties.append(0.6)
        else:
            seen_answers[normalized] = key

    if not penalties:
        return 0.0, []

    # Combine penalties: take the max but also factor in how many fired
    combined = max(penalties) * (1.0 + 0.1 * (len(penalties) - 1))
    return round(min(combined, 1.0), 3), flags


def check_logic_consistency(
    response: SurveyResponse,
    consistency_rules: list[tuple[int, int, str]] | None = None,
) -> tuple[float, list[str]]:
    """
    Generic logic-check: supply a list of (idx_a, idx_b, rule) tuples where
    rule is one of:
      "equal"      – scale_responses[idx_a] must equal scale_responses[idx_b]
      "not_equal"  – must differ
      "a_gt_b"     – a > b
      "a_lt_b"     – a < b

    Example: if Q3 asks overall satisfaction and Q7 asks likelihood to
    recommend, extreme disagreement between them is suspicious.
    """
    if not consistency_rules or not response.scale_responses:
        return 0.0, []

    violations = []
    for idx_a, idx_b, rule in consistency_rules:
        try:
            a = response.scale_responses[idx_a]
            b = response.scale_responses[idx_b]
        except IndexError:
            continue

        broken = False
        if rule == "equal" and a != b:
            broken = True
        elif rule == "not_equal" and a == b:
            broken = True
        elif rule == "a_gt_b" and not (a > b):
            broken = True
        elif rule == "a_lt_b" and not (a < b):
            broken = True

        if broken:
            violations.append(f"logic_fail(Q{idx_a}={a} vs Q{idx_b}={b}, rule={rule})")

    if not violations:
        return 0.0, []

    penalty = min(0.3 * len(violations), 1.0)
    return round(penalty, 3), violations


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class SurveyValidator:
    """
    Combines all checks into a single suspicion score.

    Parameters
    ----------
    bot_threshold        : score above which a response is classified as bot
    min_human_seconds    : fastest acceptable completion time
    max_human_seconds    : slowest before flagging as suspicious
    straightline_thresh  : fraction of identical answers before flagging
    min_open_ended_len   : minimum meaningful open-ended response length
    consistency_rules    : optional list of (idx_a, idx_b, rule) tuples
    check_weights        : override default weights per check name
    """

    DEFAULT_WEIGHTS = {
        "honeypot": 1.0,
        "completion_time": 0.8,
        "straightlining": 0.7,
        "scale_variance": 0.5,
        "open_ended": 0.6,
        "logic_consistency": 0.4,
    }

    def __init__(
        self,
        bot_threshold: float = 0.5,
        min_human_seconds: float = 30.0,
        max_human_seconds: float = 3600.0,
        straightline_thresh: float = 0.9,
        min_open_ended_len: int = 10,
        consistency_rules: list[tuple[int, int, str]] | None = None,
        check_weights: dict[str, float] | None = None,
    ):
        self.bot_threshold = bot_threshold
        self.min_human_seconds = min_human_seconds
        self.max_human_seconds = max_human_seconds
        self.straightline_thresh = straightline_thresh
        self.min_open_ended_len = min_open_ended_len
        self.consistency_rules = consistency_rules or []
        self.weights = {**self.DEFAULT_WEIGHTS, **(check_weights or {})}

    def validate(self, response: SurveyResponse) -> ValidationResult:
        all_flags: list[str] = []
        all_details: dict = {}

        checks = [
            ("honeypot",         check_honeypot(response)),
            ("completion_time",  check_completion_time(
                response, self.min_human_seconds, self.max_human_seconds
            )),
            ("straightlining",   check_straightlining(
                response, self.straightline_thresh
            )),
            ("scale_variance",   check_scale_variance(response)),
            ("open_ended",       check_open_ended(
                response, self.min_open_ended_len
            )),
            ("logic_consistency", check_logic_consistency(
                response, self.consistency_rules
            )),
        ]

        weighted_sum = 0.0
        total_weight = 0.0

        for name, (penalty, flags) in checks:
            w = self.weights.get(name, 0.5)
            weighted_sum += penalty * w
            total_weight += w
            all_flags.extend(flags)
            all_details[name] = {"penalty": penalty, "flags": flags}

        raw_score = weighted_sum / total_weight if total_weight else 0.0
        suspicion_score = round(min(raw_score, 1.0), 4)
        is_human = suspicion_score < self.bot_threshold

        return ValidationResult(
            respondent_id=response.respondent_id,
            is_human=is_human,
            suspicion_score=suspicion_score,
            flags=all_flags,
            details=all_details,
        )

    def validate_batch(
        self, responses: list[SurveyResponse]
    ) -> list[ValidationResult]:
        return [self.validate(r) for r in responses]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validator = SurveyValidator(
        bot_threshold=0.45,
        min_human_seconds=45,
        consistency_rules=[(0, 4, "a_gt_b")],  # Q0 score should exceed Q4 score
    )

    responses = [
        # Likely human
        SurveyResponse(
            respondent_id="resp_001",
            completion_time_s=312,
            scale_responses=[4, 3, 5, 2, 3, 4, 5, 3],
            open_ended={
                "Q_why": "I really enjoy the product because it saves me a lot of time.",
                "Q_improve": "Better mobile support would be great.",
            },
            honeypot_fields={"bot_trap": None},
        ),
        # Speeder bot
        SurveyResponse(
            respondent_id="resp_002",
            completion_time_s=4,
            scale_responses=[3, 3, 3, 3, 3, 3, 3, 3],
            open_ended={"Q_why": "good", "Q_improve": "good"},
            honeypot_fields={"bot_trap": None},
        ),
        # Straight-liner
        SurveyResponse(
            respondent_id="resp_003",
            completion_time_s=180,
            scale_responses=[5, 5, 5, 5, 5, 5, 5, 5],
            open_ended={"Q_why": "Great product!", "Q_improve": "Nothing."},
            honeypot_fields={"bot_trap": None},
        ),
        # Honeypot triggered + gibberish
        SurveyResponse(
            respondent_id="resp_004",
            completion_time_s=22,
            scale_responses=[1, 2, 4, 3, 1, 5, 2, 4],
            open_ended={"Q_why": "xkzqwpbvmn", "Q_improve": "zzzzzzzzzz"},
            honeypot_fields={"bot_trap": "I am a bot"},
        ),
        # Copy-paste across open-ended
        SurveyResponse(
            respondent_id="resp_005",
            completion_time_s=95,
            scale_responses=[3, 4, 2, 5, 3, 3, 4, 2],
            open_ended={
                "Q_why": "The service is excellent.",
                "Q_improve": "The service is excellent.",
            },
            honeypot_fields={"bot_trap": None},
        ),
    ]

    print("=" * 60)
    print("Survey Human Validation Results")
    print("=" * 60)
    for result in validator.validate_batch(responses):
        print(result)
    print("=" * 60)
