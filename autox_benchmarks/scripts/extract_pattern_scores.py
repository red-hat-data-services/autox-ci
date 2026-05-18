#!/usr/bin/env python3
"""Extract pattern scores from a completed KFP run."""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autorag_benchmark.config_loader import load_merged_benchmark_config
from autorag_benchmark.pattern_scores import extract_pattern_scores


def main():
    parser = argparse.ArgumentParser(description="Extract pattern scores from S3 for a KFP run")
    parser.add_argument("run_id", help="KFP run ID")
    parser.add_argument(
        "--config",
        default="config/benchmark.yaml",
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--credentials",
        default="config/credentials.ini",
        help="Path to credentials INI file",
    )
    parser.add_argument(
        "--output-json",
        help="Save pattern scores to JSON file",
    )
    parser.add_argument(
        "--bucket",
        default="ai-eng-cracow",
        help="S3 bucket name (default: ai-eng-cracow)",
    )
    args = parser.parse_args()

    # Load config
    cfg, _ = load_merged_benchmark_config(
        Path(args.config).resolve(),
        Path(args.credentials).resolve() if args.credentials else None,
    )

    print(f"Extracting pattern scores for run: {args.run_id}")
    print("=" * 80)

    # Extract pattern scores
    pattern_scores = extract_pattern_scores(
        run_id=args.run_id,
        config=cfg,
        bucket=args.bucket,
    )

    # Check for errors
    if "error" in pattern_scores:
        print(f"\n❌ Error: {pattern_scores['error']}")
        sys.exit(1)

    # Display results
    print(f"\nS3 Location: {pattern_scores.get('rag_patterns_s3_uri', 'unknown')}")
    print(f"Number of patterns: {pattern_scores.get('num_patterns', 0)}")

    if "best_pattern" in pattern_scores:
        best = pattern_scores["best_pattern"]
        print(f"\n🏆 Best Pattern: {best.get('name')}")
        print(f"   Final Score: {best.get('final_score')}")
        print(f"   Scores: {json.dumps(best.get('scores', {}), indent=6)}")

    print("\n" + "=" * 80)
    print("\nAll Patterns (sorted by final_score):\n")

    for i, pattern in enumerate(pattern_scores.get("patterns", []), 1):
        print(f"{i}. {pattern.get('pattern_name', 'unknown')}")
        print(f"   Final Score: {pattern.get('final_score', 'N/A')}")
        print(f"   Execution Time: {pattern.get('execution_time', 'N/A')}s")

        scores = pattern.get("scores", {})
        if scores:
            print(f"   Scores:")
            for metric, value in scores.items():
                print(f"     - {metric}: {value}")

        if "error" in pattern:
            print(f"   ⚠️  Error: {pattern['error']}")

        print()

    # Save to JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(pattern_scores, indent=2))
        print(f"✓ Saved pattern scores to {output_path}")


if __name__ == "__main__":
    main()
