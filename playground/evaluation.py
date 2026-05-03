import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# ─── Regex Banks (from §8.2, §8.4 of evaluation framework) ───

# D1: Voice Fidelity Checks
EMOJI_REGEX = re.compile(
    r'[\u263a-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF\u2702-\u27B0\u24C2-\U0001F251]'
)
AI_SELF_REVEAL_REGEX = re.compile(
    r'\b(I am an AI|I am a language model|I don\'t have feelings|as an assistant)\b',
    re.IGNORECASE
)
MODERN_COACH_REGEX = re.compile(
    r'\b(hey there|hi there|how can I help you|I understand how you feel|that\'s a great question)\b',
    re.IGNORECASE
)

# D2: Character-Specific Time Anchor Checks (Oppenheimer example)
TIME_ANCHOR_REGEX = {
    "oppenheimer": re.compile(
        r'\b(after the (war|project|test)|grim conclusion|hearings|lost my Q|McCarthy|Princeton lecture|after Los Alamos|once it was over)\b',
        re.IGNORECASE
    )
}

# D5/D8: Pattern Detection Regex (§8.4)
QUESTION_STACK_REGEX = re.compile(r'\?[^.?]{0,30}\?[^.?]{0,30}\?')  # 3+ questions
THERAPY_SPIRAL_REGEX = re.compile(
    r'(how does that feel|what is it about .* that|tell me, when you consider)\?\s*$',
    re.IGNORECASE
)
ECHO_TRAP_OPENER_REGEX = re.compile(r'^\s*[^.!?]{1,15}\.\s*Yes[.,]\s+', re.IGNORECASE)
COACH_SLIP_REGEX = re.compile(
    r'\b(what (are|is|would|do) you (hope|hoping|want|imagine|think|feel|consider)|tell me, what|what (would|might) (it|that) (look|feel|be) like|help (us|you) find)\b',
    re.IGNORECASE
)

# ─── Data Models ───

@dataclass
class TurnEvaluation:
    turn_num: int
    user_message: str
    assistant_reply: str
    character_id: str
    model_name: str

    # Automated Checks (D1-D3)
    auto_checks: Dict = field(default_factory=dict)
    # Dimension Scores (D1-D8: 0/1/2, D8 n/a for turn 1)
    dim_scores: Dict = field(default_factory=dict)
    # Pattern Hits (§8.4)
    pattern_hits: List[str] = field(default_factory=list)
    # Total per-turn score (max 16)
    total_score: int = 0

@dataclass
class SessionEvaluation:
    session_id: str
    tenant_id: str = "default"  # For future multi-tenant
    character_id: str = ""
    model_name: str = ""
    turn_evals: List[TurnEvaluation] = field(default_factory=list)

    def session_total(self) -> int:
        """Max 48 for 4-turn, 96 for 6-turn session"""
        return sum(t.total_score for t in self.turn_evals)

    def dimension_averages(self) -> Dict[str, float]:
        dims = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
        totals = {d: 0 for d in dims}
        counts = {d: 0 for d in dims}
        for t in self.turn_evals:
            for d in dims:
                if d in t.dim_scores and t.dim_scores[d] != "n/a":
                    totals[d] += t.dim_scores[d]
                    counts[d] += 1
        return {d: round(totals[d]/counts[d], 2) if counts[d] > 0 else 0 for d in dims}

    def pattern_summary(self) -> Dict[str, int]:
        summary = {}
        for t in self.turn_evals:
            for p in t.pattern_hits:
                summary[p] = summary.get(p, 0) + 1
        return summary

# ─── Core Evaluation Logic ───

class EvaluationEngine:
    def __init__(self, character_id: str, enable_eval: bool = True):
        self.character_id = character_id
        self.enable_eval = enable_eval

    def evaluate_turn(self, turn_num: int, user_msg: str, assistant_reply: str,
                     model_name: str, prior_turn: Optional[TurnEvaluation] = None) -> Optional[TurnEvaluation]:
        """Evaluate a single turn (D1-D8 + pattern detection)"""
        if not self.enable_eval:
            return None

        turn_eval = TurnEvaluation(
            turn_num=turn_num,
            user_message=user_msg,
            assistant_reply=assistant_reply,
            character_id=self.character_id,
            model_name=model_name
        )

        # Run all checks
        self._run_auto_checks(turn_eval)
        self._score_dimensions(turn_eval, prior_turn)
        self._detect_patterns(turn_eval, prior_turn)

        # Calculate total score
        turn_eval.total_score = sum(
            v for v in turn_eval.dim_scores.values() if isinstance(v, int)
        )
        return turn_eval

    def _run_auto_checks(self, te: TurnEvaluation):
        """D1-D3 automated checks (§8.3 Step 1)"""
        reply = te.assistant_reply
        words = reply.split()
        sentences = [s for s in re.split(r'[.!?]', reply) if s.strip()]

        # D1: Emoji, AI self-reveal, modern coach
        emoji_hits = len(EMOJI_REGEX.findall(reply))
        self_reveal_hits = len(AI_SELF_REVEAL_REGEX.findall(reply))
        coach_hits = len(MODERN_COACH_REGEX.findall(reply))
        banned_hits = emoji_hits + self_reveal_hits + coach_hits

        # D2: Time anchor drift (character-specific)
        time_drift_hits = 0
        if self.character_id in TIME_ANCHOR_REGEX:
            time_drift_hits = len(TIME_ANCHOR_REGEX[self.character_id].findall(reply))

        # D3: Length
        word_count = len(words)
        sentence_count = len(sentences)
        exclaim_count = reply.count("!")

        te.auto_checks = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "exclaim_count": exclaim_count,
            "emoji_hits": emoji_hits,
            "banned_hits": banned_hits,
            "time_drift_hits": time_drift_hits,
            "self_reveal_hits": self_reveal_hits,
            "coach_hits": coach_hits
        }

    def _score_dimensions(self, te: TurnEvaluation, prior_turn: Optional[TurnEvaluation]):
        """Score D1-D8 per §8.2 rubric"""
        checks = te.auto_checks
        reply = te.assistant_reply
        user_msg = te.user_message
        words = reply.split()
        sentences = [s for s in re.split(r'[.!?]', reply) if s.strip()]

        # D1: Voice Fidelity
        if checks["emoji_hits"] > 0 or checks["self_reveal_hits"] > 0:
            te.dim_scores["D1"] = 0
        elif checks["banned_hits"] > 0 or checks["coach_hits"] > 0:
            te.dim_scores["D1"] = 1
        else:
            # Check for register markers (character-specific, simplified for MVP)
            te.dim_scores["D1"] = 2 if "I " in reply[:50] else 1

        # D2: Scene Anchoring
        if checks["time_drift_hits"] > 0:
            te.dim_scores["D2"] = 0  # Time anchor drift
        elif self.character_id == "oppenheimer" and "1944" not in reply and "Los Alamos" not in reply:
            te.dim_scores["D2"] = 1  # Generic but correct beat
        else:
            te.dim_scores["D2"] = 2  # Concrete details (simplified for MVP)

        # D3: Length Discipline
        if checks["word_count"] > 90 or checks["sentence_count"] > 4:
            te.dim_scores["D3"] = 0
        elif checks["word_count"] > 60 or checks["sentence_count"] > 3:
            te.dim_scores["D3"] = 1
        else:
            te.dim_scores["D3"] = 2

        # D4: User Responsiveness (simplified: check if user keywords appear in reply)
        user_keywords = [w.lower() for w in user_msg.split() if len(w) > 3]
        reply_lower = reply.lower()
        keyword_matches = sum(1 for kw in user_keywords if kw in reply_lower)
        if keyword_matches == 0:
            te.dim_scores["D4"] = 0
        elif keyword_matches <= 2:
            te.dim_scores["D4"] = 1
        else:
            te.dim_scores["D4"] = 2

        # D5: Question Hygiene
        question_count = reply.count("?")
        if question_count >= 3 or QUESTION_STACK_REGEX.search(reply):
            te.dim_scores["D5"] = 0
        elif question_count == 2:
            te.dim_scores["D5"] = 1
        else:
            te.dim_scores["D5"] = 2

        # D6: Practical Service (simplified: check for concrete action verbs)
        action_verbs = re.findall(r'\b(type|run|write|delete|install|open|click)\b', reply_lower)
        if COACH_SLIP_REGEX.search(reply) and len(action_verbs) == 0:
            te.dim_scores["D6"] = 0
        elif len(action_verbs) > 0:
            te.dim_scores["D6"] = 2
        else:
            te.dim_scores["D6"] = 1

        # D7: Dual-Flow Balance (simplified: check character vs user content ratio)
        char_words = len([w for w in words if w.istitle() or "I " in w])  # Rough character content
        user_words = len(user_msg.split())
        ratio = char_words / max(len(words), 1)
        if "stuck" in user_msg.lower() or "lost" in user_msg.lower():
            # User stuck: target ~30% char, 70% user
            te.dim_scores["D7"] = 2 if 0.2 <= ratio <= 0.4 else 1 if 0.1 <= ratio <= 0.5 else 0
        else:
            # User working: target ~70% char, 30% user
            te.dim_scores["D7"] = 2 if 0.6 <= ratio <= 0.8 else 1 if 0.5 <= ratio <= 0.9 else 0

        # D8: Anti-Doom-Loop (only turn 2+)
        if te.turn_num == 1:
            te.dim_scores["D8"] = "n/a"
        elif prior_turn:
            # Check if reply is structurally similar to prior reply
            prior_words = prior_turn.assistant_reply.split()
            overlap = len(set(words) & set(prior_words)) / max(len(words), 1)
            if overlap > 0.7:
                te.dim_scores["D8"] = 0
            elif overlap > 0.4:
                te.dim_scores["D8"] = 1
            else:
                te.dim_scores["D8"] = 2
        else:
            te.dim_scores["D8"] = "n/a"

    def _detect_patterns(self, te: TurnEvaluation, prior_turn: Optional[TurnEvaluation]):
        """Detect §8.4 failure patterns"""
        reply = te.assistant_reply
        user_msg = te.user_message

        # 8.4.1 Therapy Spiral
        if THERAPY_SPIRAL_REGEX.search(reply):
            te.pattern_hits.append("Therapy Spiral")

        # 8.4.2 Memoir Dump
        if te.auto_checks["sentence_count"] >= 4 or te.auto_checks["word_count"] > 90:
            te.pattern_hits.append("Memoir Dump")

        # 8.4.3 Question Stack
        if QUESTION_STACK_REGEX.search(reply):
            te.pattern_hits.append("Question Stack")

        # 8.4.4 Echo Trap
        if prior_turn and te.turn_num >= 2:
            user_stuck = any(w in user_msg.lower() for w in ["stuck", "lost", "confused"])
            prior_user_stuck = any(w in prior_turn.user_message.lower() for w in ["stuck", "lost", "confused"])
            if user_stuck and prior_user_stuck and ECHO_TRAP_OPENER_REGEX.search(reply):
                te.pattern_hits.append("Echo Trap")

        # 8.4.5 Coach Slip
        if COACH_SLIP_REGEX.search(reply):
            te.pattern_hits.append("Coach Slip")

        # 8.4.6 Time Anchor Drift
        if te.auto_checks["time_drift_hits"] > 0:
            te.pattern_hits.append("Time Anchor Drift")

        # 8.4.7 3-Stack Opener
        if ECHO_TRAP_OPENER_REGEX.search(reply):
            te.pattern_hits.append("3-Stack Opener")
