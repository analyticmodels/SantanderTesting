"""
Evaluate hallucination questions and answers using DeepEval metrics.

This script loads question-answer pairs from hallucinationQuestions_all.parquet
and evaluates them using appropriate DeepEval metrics.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    GEval,
)

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "hallucinationQuestions_all.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "evaluation_results.parquet"
SAMPLE_SIZE = None  # Set to a number to limit evaluation, None for all


def load_data(path: Path, sample_size: int = None) -> pd.DataFrame:
    """Load question-answer pairs from parquet file."""
    df = pd.read_parquet(path)
    if sample_size:
        df = df.head(sample_size)
    print(f"Loaded {len(df)} question-answer pairs")
    return df


def create_test_cases(df: pd.DataFrame) -> list[LLMTestCase]:
    """Create LLMTestCase objects from dataframe."""
    test_cases = []
    for _, row in df.iterrows():
        # For hallucination detection, we use the question as input
        # and the answer as actual_output
        # Since we don't have retrieval context, we'll use the question as context
        # for faithfulness evaluation
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            # Using question as context since these are Q&A pairs without retrieval
            retrieval_context=[row["question"]],
        )
        test_cases.append(test_case)
    return test_cases


def initialize_metrics() -> list:
    """Initialize all relevant DeepEval metrics."""
    metrics = []

    # Answer Relevancy - measures if the answer addresses the question
    metrics.append(
        AnswerRelevancyMetric(
            threshold=0.5,
            model="gpt-4o-mini",
            include_reason=True,
        )
    )

    # Faithfulness - measures if claims in the answer are grounded
    metrics.append(
        FaithfulnessMetric(
            threshold=0.5,
            model="gpt-4o-mini",
            include_reason=True,
        )
    )

    # Hallucination - specifically detects hallucinated content
    metrics.append(
        HallucinationMetric(
            threshold=0.5,
            model="gpt-4o-mini",
            include_reason=True,
        )
    )

    # Toxicity - checks for harmful content
    metrics.append(
        ToxicityMetric(
            threshold=0.5,
            model="gpt-4o-mini",
            include_reason=True,
        )
    )

    # Bias - checks for biased content
    metrics.append(
        BiasMetric(
            threshold=0.5,
            model="gpt-4o-mini",
            include_reason=True,
        )
    )

    # Custom G-Eval for answer quality
    metrics.append(
        GEval(
            name="Answer Quality",
            criteria="Evaluate the quality and helpfulness of the answer. Consider clarity, completeness, and accuracy.",
            evaluation_params=["input", "actual_output"],
            model="gpt-4o-mini",
            threshold=0.5,
        )
    )

    return metrics


def evaluate_single(test_case: LLMTestCase, metrics: list) -> dict:
    """Evaluate a single test case with all metrics."""
    results = {
        "question": test_case.input,
        "answer": test_case.actual_output,
    }

    for metric in metrics:
        try:
            metric.measure(test_case)
            metric_name = metric.__class__.__name__.replace("Metric", "")
            results[f"{metric_name}_score"] = metric.score
            results[f"{metric_name}_reason"] = metric.reason if hasattr(metric, "reason") else None
            results[f"{metric_name}_passed"] = metric.score >= metric.threshold
        except Exception as e:
            metric_name = metric.__class__.__name__.replace("Metric", "")
            results[f"{metric_name}_score"] = None
            results[f"{metric_name}_reason"] = f"Error: {str(e)}"
            results[f"{metric_name}_passed"] = None

    return results


def run_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Run evaluation on all question-answer pairs."""
    test_cases = create_test_cases(df)
    metrics = initialize_metrics()

    print(f"Evaluating {len(test_cases)} test cases with {len(metrics)} metrics...")
    print("Metrics:", [m.__class__.__name__ for m in metrics])

    results = []
    for test_case in tqdm(test_cases, desc="Evaluating"):
        result = evaluate_single(test_case, metrics)
        results.append(result)

    return pd.DataFrame(results)


def main():
    """Main entry point."""
    print("=" * 60)
    print("DeepEval Hallucination Questions Evaluation")
    print("=" * 60)

    # Load data
    df = load_data(DATA_PATH, SAMPLE_SIZE)

    # Run evaluation
    results_df = run_evaluation(df)

    # Save results
    results_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    score_columns = [col for col in results_df.columns if col.endswith("_score")]
    for col in score_columns:
        metric_name = col.replace("_score", "")
        scores = results_df[col].dropna()
        if len(scores) > 0:
            print(f"\n{metric_name}:")
            print(f"  Mean:   {scores.mean():.3f}")
            print(f"  Median: {scores.median():.3f}")
            print(f"  Min:    {scores.min():.3f}")
            print(f"  Max:    {scores.max():.3f}")

            passed_col = f"{metric_name}_passed"
            if passed_col in results_df.columns:
                passed = results_df[passed_col].sum()
                total = results_df[passed_col].notna().sum()
                print(f"  Passed: {passed}/{total} ({100*passed/total:.1f}%)")


if __name__ == "__main__":
    main()
