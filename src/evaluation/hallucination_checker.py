"""
Hallucination Checker & SCQ Composite Metric Module
Verifies factual claim support (SUPPORTED / CONTRADICTED / UNKNOWN) and computes composite SCQ score.

Verification backends:
- "nli": a cross-encoder NLI model (config ``models.nli_verification.model_name``) if it can be
  loaded locally. Not available in offline environments.
- "lexical" (default fallback): content-word support plus a numeric-conflict check.
  SUPPORTED   = enough of the claim's content words appear in the evidence and every number
                in the claim also appears in the evidence.
  CONTRADICTED = the claim states a number that the evidence does not contain while the evidence
                 does report numbers (a concrete numeric conflict).
  UNKNOWN     = anything else. Low lexical overlap is *not* evidence of contradiction.
"""

import re
from typing import Dict, Any, List, Optional

_STOPWORDS = {
    "this", "that", "with", "from", "were", "which", "their", "there", "these", "those",
    "have", "been", "into", "than", "then", "also", "each", "over", "under", "across",
    "shows", "show", "figure", "plot", "plots", "visualization", "the", "and", "for",
}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _content_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-\+]*", text.lower())
    return [w for w in words if len(w) > 3 and w not in _STOPWORDS]


class HallucinationChecker:
    def __init__(self, config: Dict[str, Any] = None, support_threshold: float = 0.5, use_nli: bool = True):
        self.config = config or {}
        self.scq_weights = self.config.get("scq_evaluation.weights", None) or {
            "w_entity_coverage": 0.30,
            "w_technical_correctness": 0.30,
            "w_reconstruction_fidelity": 0.20,
            "w_retrieval_accuracy": 0.20
        }
        self.support_threshold = support_threshold
        self._nli = self._load_nli() if use_nli else None
        self.backend = "nli" if self._nli is not None else "lexical"

    def _load_nli(self) -> Optional[Any]:
        model_name = self.config.get("models.nli_verification.model_name", None)
        if not model_name:
            return None
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(model_name)
        except Exception:
            return None

    def _verify_claim_lexical(self, claim: str, evidence_text: str) -> str:
        claim_words = _content_words(claim)
        claim_nums = set(_NUM_RE.findall(claim))
        evid_nums = set(_NUM_RE.findall(evidence_text))
        if claim_nums and evid_nums and not claim_nums.issubset(evid_nums):
            return "CONTRADICTED"
        if not claim_words:
            return "UNKNOWN"
        support = sum(1 for w in claim_words if w in evidence_text) / len(claim_words)
        if support >= self.support_threshold and claim_nums.issubset(evid_nums):
            return "SUPPORTED"
        return "UNKNOWN"

    def _verify_claim_nli(self, claim: str, evidence_text: str) -> str:
        # Label order for cross-encoder/nli-* models: contradiction, entailment, neutral
        scores = self._nli.predict([(evidence_text, claim)])[0]
        label = ["CONTRADICTED", "SUPPORTED", "UNKNOWN"][int(scores.argmax())]
        return label

    def verify_caption(self, caption: str, retrieved_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifies claims in generated caption against retrieved paper evidence.
        """
        claims = [c.strip() for c in re.split(r"(?<!\d)\.(?!\d)", caption) if c.strip()]
        evidence_text = " ".join([e.get("text", "") for e in retrieved_evidence]).lower()

        verification_results = []
        supported_count = 0
        for claim in claims:
            if self._nli is not None and evidence_text:
                status = self._verify_claim_nli(claim, evidence_text)
            else:
                status = self._verify_claim_lexical(claim, evidence_text)
            if status == "SUPPORTED":
                supported_count += 1
            verification_results.append({
                "claim": claim,
                "evidence_snippet": evidence_text[:100] + "...",
                "status": status
            })

        correctness_ratio = supported_count / max(len(claims), 1)

        backend_desc = "NLI cross-encoder" if self.backend == "nli" else "lexical support + numeric-conflict check (no NLI model loaded)"
        return {
            "claims_verification": verification_results,
            "technical_correctness_ratio": float(correctness_ratio),
            "verification_backend": self.backend,
            "status": f"PROTOTYPE ({backend_desc})"
        }

    def compute_scq_metric(
        self,
        entity_coverage: float,
        technical_correctness: float,
        reconstruction_fidelity: float,
        retrieval_accuracy: float
    ) -> Dict[str, Any]:
        """
        Calculates composite Scientific Caption Quality (SCQ) metric:
        SCQ = w_1 * Entity_Coverage + w_2 * Technical_Correctness + w_3 * Reconstruction_Fidelity + w_4 * Retrieval_Accuracy
        """
        w1 = self.scq_weights.get("w_entity_coverage", 0.30)
        w2 = self.scq_weights.get("w_technical_correctness", 0.30)
        w3 = self.scq_weights.get("w_reconstruction_fidelity", 0.20)
        w4 = self.scq_weights.get("w_retrieval_accuracy", 0.20)

        scq_score = (
            w1 * entity_coverage +
            w2 * technical_correctness +
            w3 * reconstruction_fidelity +
            w4 * retrieval_accuracy
        )

        return {
            "scq_composite_score": float(scq_score),
            "breakdown": {
                "entity_coverage": entity_coverage,
                "technical_correctness": technical_correctness,
                "reconstruction_fidelity": reconstruction_fidelity,
                "retrieval_accuracy": retrieval_accuracy
            },
            "weights_used": {
                "w_entity_coverage": w1,
                "w_technical_correctness": w2,
                "w_reconstruction_fidelity": w3,
                "w_retrieval_accuracy": w4
            }
        }
