import os
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(dotenv_path=dotenv_path)

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable

# DeepEval imports
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.metrics.g_eval import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from groq import Groq

logger = logging.getLogger(__name__)

# --- Configuration Constants (Self-contained config replacement) ---
DEEPEVAL_JUDGE_MODEL = os.getenv("GROQ_EVALUATION_MODEL", "openai/gpt-oss-20b")
DEEPEVAL_TIMEOUT_S = 30.0
REASONING_EFFORT = None  # reasoning effort configuration (None for standard models)
DEEPEVAL_MAX_RETRIES = 5
DEEPEVAL_MAX_RETRY_WAIT = 60.0
DEEPEVAL_MAX_TOKENS_CAP = 4096

# Parse Gemini/Groq's "Please retry in 19.9s" / "Please try again in 7m48.72s" hint.
_RETRY_AFTER_RE = re.compile(r"(?:try again|retry) in (?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)


def _retry_after_seconds(message: str) -> float | None:
    m = _RETRY_AFTER_RE.search(message or "")
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


class GroqJudge(DeepEvalBaseLLM):
    """A DeepEval judge model that calls Groq under the hood."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or DEEPEVAL_JUDGE_MODEL
        super().__init__(self._model_name)

    def load_model(self):
        from groq import Groq
        key = os.getenv("EVALUATION_API_KEY") or os.getenv("GROQ_API_KEY")
        if not key or key in ("your_groq_api_key_here", "your_evaluation_api_key_here"):
            raise RuntimeError(
                "No Groq API key set for DeepEval judge calls. Add EVALUATION_API_KEY "
                "(or GROQ_API_KEY) to your .env file."
            )
        return Groq(api_key=key)

    def get_model_name(self) -> str:
        return self._model_name

    def _call(self, prompt: str, schema=None) -> str:
        for attempt in range(DEEPEVAL_MAX_RETRIES + 1):
            try:
                response = self.model.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"} if schema is not None else None
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower() or "resource exhausted" in msg.lower()
                is_transient = "503" in msg or "unavailable" in msg.lower() or "high demand" in msg.lower() or "504" in msg or "gateway timeout" in msg.lower() or "502" in msg or "bad gateway" in msg.lower()
                
                if (is_rate_limit or is_transient) and attempt < DEEPEVAL_MAX_RETRIES:
                    wait = _retry_after_seconds(msg) if is_rate_limit else (2.0 ** attempt + 2.0)
                    wait = wait or 10.0
                    if wait <= DEEPEVAL_MAX_RETRY_WAIT:
                        logger.warning(f"[JUDGE RETRY] Rate limit/transient error ({msg}). Retrying in {wait:.2f}s... (Attempt {attempt+1}/{DEEPEVAL_MAX_RETRIES})")
                        time.sleep(wait)
                        continue
                logger.warning("DeepEval Groq judge call failed", exc_info=True)
                raise

    def generate(self, prompt: str, schema=None):
        raw = self._call(prompt, schema=schema)
        if schema is None:
            return raw
        return schema.model_validate_json(raw)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


@traceable(name="evaluate_live_metrics", run_type="chain")
def evaluate(question: str, answer: str, contexts: list[str]) -> dict:
    """Score one answer on the four reference-free DeepEval metrics.

    `contexts` are the full text of the retrieved chunks.
    """
    if not contexts:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_relevancy": None,
        }

    # Initialize model
    judge_model = GroqJudge()
    
    # Initialize metrics
    faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=judge_model)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=judge_model)
    precision_metric = ContextualPrecisionMetric(threshold=0.5, model=judge_model)
    context_relevancy_metric = ContextualRelevancyMetric(threshold=0.5, model=judge_model)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=answer,  # Use answer as reference proxy for ContextualPrecisionMetric
        retrieval_context=contexts
    )
    
    def run_metric(metric, case):
        try:
            metric.measure(case)
            return _round(metric.score)
        except Exception as e:
            logger.exception(f"DeepEval {metric.__class__.__name__} failed: {e}")
            return None

    # Run metrics concurrently in parallel (no longer sequential, no artificial sleeps)
    with ThreadPoolExecutor(max_workers=4) as pool:
        faith_future = pool.submit(run_metric, faithfulness_metric, test_case)
        rel_future = pool.submit(run_metric, relevancy_metric, test_case)
        prec_future = pool.submit(run_metric, precision_metric, test_case)
        ctx_rel_future = pool.submit(run_metric, context_relevancy_metric, test_case)
        
        faith = faith_future.result()
        rel = rel_future.result()
        prec = prec_future.result()
        ctx_rel = ctx_rel_future.result()
        
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "context_precision": prec,
        "context_relevancy": ctx_rel,
    }


@traceable(name="evaluate_with_ground_truth", run_type="chain")
def evaluate_with_ground_truth(
    question: str, answer: str, ground_truth: str, contexts: list[str]
) -> dict:
    """Score one answer on all six DeepEval metrics concurrently, using a reference ground truth."""
    if not contexts:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_relevance": None,
            "context_recall": None,
            "answer_correctness": None,
        }
        
    judge_model = GroqJudge()
    
    # Initialize all 6 metrics
    faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=judge_model)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=judge_model)
    precision_metric = ContextualPrecisionMetric(threshold=0.5, model=judge_model)
    context_relevancy_metric = ContextualRelevancyMetric(threshold=0.5, model=judge_model)
    recall_metric = ContextualRecallMetric(threshold=0.5, model=judge_model)
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the actual output is factually correct and complete based on the expected output.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge_model
    )
    
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=contexts
    )
    
    def run_metric(metric, case):
        try:
            metric.measure(case)
            return _round(metric.score)
        except Exception as e:
            logger.exception(f"DeepEval {metric.__class__.__name__} failed: {e}")
            return None
            
    # Run all 6 metrics in parallel concurrently
    with ThreadPoolExecutor(max_workers=6) as pool:
        faith_future = pool.submit(run_metric, faithfulness_metric, test_case)
        rel_future = pool.submit(run_metric, relevancy_metric, test_case)
        prec_future = pool.submit(run_metric, precision_metric, test_case)
        ctx_rel_future = pool.submit(run_metric, context_relevancy_metric, test_case)
        recall_future = pool.submit(run_metric, recall_metric, test_case)
        correctness_future = pool.submit(run_metric, correctness_metric, test_case)
        
        faith = faith_future.result()
        rel = rel_future.result()
        prec = prec_future.result()
        ctx_rel = ctx_rel_future.result()
        recall = recall_future.result()
        correctness = correctness_future.result()
        
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "context_precision": prec,
        "context_relevance": ctx_rel,  # maps to schema mapped key
        "context_recall": recall,
        "answer_correctness": correctness,
    }
