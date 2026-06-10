#!/usr/bin/env python3
"""
Generate RAG benchmark datasets (BEIR, OpenRAGBench) and optionally upload to S3.

This script generates knowledge base documents and benchmark Q&A pairs for use with
the AutoRAG benchmark pipeline. Generated datasets can be stored locally or uploaded
to S3 for use in the benchmarking workflow.

Usage:
  # Generate OpenRAGBench locally
  python scripts/generate_rag_datasets.py --dataset open_ragbench --num-samples 10 \
    --output-dir ./generated_datasets/open_ragbench

  # Generate BEIR scifact locally
  python scripts/generate_rag_datasets.py --dataset beir --beir-dataset scifact \
    --num-samples 50 --output-dir ./generated_datasets/beir_scifact

  # Generate and upload to S3 using .env (recommended)
  # First, copy .env.example to .env with AWS_* keys:
  #   [s3]
  #   endpoint = https://s3.amazonaws.com
  #   aws_access_key_id = YOUR_KEY
  #   aws_secret_access_key = YOUR_SECRET
  #   aws_default_region = us-east-1

  # Upload to: s3://<your-bucket>/datasets/rag/open_ragbench/arxiv/txt/50/
  # Local output: ./generated_datasets/open_ragbench/txt/
  python scripts/generate_rag_datasets.py --dataset open_ragbench --num-samples 50 \
    --upload-to-s3

  # Use custom credentials file
  python scripts/generate_rag_datasets.py --dataset beir --beir-dataset scifact \
    --num-samples 100 --upload-to-s3 --env-file /path/to/.env

  # Or use environment variables (fallback)
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  python scripts/generate_rag_datasets.py --dataset open_ragbench --num-samples 50 \
    --upload-to-s3

Requirements:
  pip install -e ".[datasets]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path so we can import autorag_benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAG benchmark datasets and optionally upload to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["beir", "open_ragbench", "slidevqa", "html_rag", "nomiracl", "mlqa", "mkqa"],
        help="Dataset to generate",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local output directory (default: ./generated_datasets/{dataset_name}/{format})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)",
    )
    parser.add_argument(
        "--output-format",
        choices=["txt", "md", "html", "pdf", "pptx", "png", "jpg"],
        default="txt",
        help="Output format for knowledge base documents (default: txt)",
    )

    # BEIR-specific options
    parser.add_argument(
        "--beir-dataset",
        default="scifact",
        help="BEIR dataset name (e.g., scifact, nfcorpus, nq) - only for --dataset beir (default: scifact)",
    )
    parser.add_argument(
        "--beir-split",
        default="test",
        help="BEIR dataset split (e.g., train, test, dev) - only for --dataset beir (default: test)",
    )

    # SlideVQA-specific options
    parser.add_argument(
        "--slidevqa-split",
        default="val",
        help="SlideVQA dataset split (train, val, test) - only for --dataset slidevqa (default: val)",
    )

    # HtmlRag (Natural Questions) specific options
    parser.add_argument(
        "--nq-split",
        default="validation",
        help="Natural Questions split (train, validation) - only for --dataset html_rag (default: validation)",
    )

    # NoMIRACL specific options
    parser.add_argument(
        "--nomiracl-language",
        default="en",
        help="NoMIRACL language code (ar, bn, de, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, yo, zh) "
             "- only for --dataset nomiracl (default: en)",
    )
    parser.add_argument(
        "--nomiracl-split",
        default="dev",
        help="NoMIRACL split (dev, test) - only for --dataset nomiracl (default: dev)",
    )

    # MLQA specific options
    parser.add_argument(
        "--mlqa-language",
        default="en",
        help="MLQA language code (ar, de, en, es, hi, vi, zh) "
             "- only for --dataset mlqa (default: en)",
    )
    parser.add_argument(
        "--mlqa-split",
        default="test",
        help="MLQA split (test, validation) - only for --dataset mlqa (default: test)",
    )

    # MKQA specific options
    parser.add_argument(
        "--mkqa-language",
        default="en",
        help="MKQA language code (ar, da, de, en, es, fi, fr, he, hu, it, ja, km, ko, "
             "ms, nl, no, pl, pt, ru, sv, th, tr, vi, zh_cn, zh_hk, zh_tw) "
             "- only for --dataset mkqa (default: en)",
    )

    # S3 upload options
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload generated dataset to S3 after generation",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to .env file (default: project .env)",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="S3 bucket for upload (default: BENCHMARK_INPUT_DATA_BUCKET_NAME from .env)",
    )
    parser.add_argument(
        "--s3-prefix",
        help="S3 key prefix (default: datasets/rag/{dataset_type}/{variant}/{num_samples})",
    )

    args = parser.parse_args()

    # Import dataset registry
    from autorag_benchmark.datasets import get, list_datasets

    try:
        prepare_fn, default_options = get(args.dataset)
    except KeyError:
        print(f"Error: Unknown dataset '{args.dataset}'", file=sys.stderr)
        print(f"Available datasets: {', '.join(list_datasets())}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory (include format to prevent conflicts)
    if args.output_dir is None:
        dataset_name = args.dataset
        if args.dataset == "beir":
            dataset_name = f"beir_{args.beir_dataset}"
        elif args.dataset == "slidevqa":
            dataset_name = f"slidevqa_{args.slidevqa_split}"
        elif args.dataset == "html_rag":
            dataset_name = f"html_rag_{args.nq_split}"
        elif args.dataset == "nomiracl":
            dataset_name = f"nomiracl_{args.nomiracl_language}_{args.nomiracl_split}"
        elif args.dataset == "mlqa":
            dataset_name = f"mlqa_{args.mlqa_language}_{args.mlqa_split}"
        elif args.dataset == "mkqa":
            dataset_name = f"mkqa_{args.mkqa_language}"
        args.output_dir = Path("./generated_datasets") / dataset_name / args.output_format

    kb_dir = args.output_dir / "knowledge_base"
    bench_path = args.output_dir / "benchmark_data.json"

    # Build options for prepare function
    options = dict(default_options)
    options["num_samples"] = args.num_samples
    options["output_format"] = args.output_format

    if args.dataset == "beir":
        options["beir_dataset"] = args.beir_dataset
        options["split"] = args.beir_split
    elif args.dataset == "slidevqa":
        options["split"] = args.slidevqa_split
    elif args.dataset == "html_rag":
        options["split"] = args.nq_split
    elif args.dataset == "nomiracl":
        options["language"] = args.nomiracl_language
        options["split"] = args.nomiracl_split
    elif args.dataset == "mlqa":
        options["language"] = args.mlqa_language
        options["split"] = args.mlqa_split
    elif args.dataset == "mkqa":
        options["language"] = args.mkqa_language

    # Generate dataset
    print(f"\n{'='*60}")
    print(f"Generating {args.dataset} dataset")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output format: {args.output_format}")
    if args.dataset == "beir":
        print(f"BEIR dataset: {args.beir_dataset}")
        print(f"BEIR split: {args.beir_split}")
    elif args.dataset == "slidevqa":
        print(f"SlideVQA split: {args.slidevqa_split}")
    elif args.dataset == "html_rag":
        print(f"NQ split: {args.nq_split}")
    elif args.dataset == "nomiracl":
        print(f"Language: {args.nomiracl_language}")
        print(f"Split: {args.nomiracl_split}")
    elif args.dataset == "mlqa":
        print(f"Language: {args.mlqa_language}")
        print(f"Split: {args.mlqa_split}")
    elif args.dataset == "mkqa":
        print(f"Language: {args.mkqa_language}")
    print()

    try:
        num_docs, num_entries = prepare_fn(kb_dir, bench_path, **options)
    except Exception as e:
        print(f"\nError generating dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"{'='*60}")
    print(f"Documents: {num_docs}")
    print(f"Benchmark entries: {num_entries}")
    print(f"Knowledge base: {kb_dir}")
    print(f"Benchmark data: {bench_path}")
    print()

    # Upload to S3 if requested
    input_data_key = None
    test_data_key = None

    if args.upload_to_s3:
        # Import S3 utilities
        from autorag_benchmark.s3_dataset_upload import (
            upload_dataset_to_s3,
            ensure_s3_bucket_exists,
            get_s3_boto_config,
        )
        from autorag_benchmark.storage_buckets import resolve_dataset_upload_bucket_from_env

        try:
            upload_bucket = resolve_dataset_upload_bucket_from_env(args.env_file, args.s3_bucket)
        except ValueError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            sys.exit(1)

        s3_config = get_s3_boto_config(args.env_file)
        if not s3_config:
            print("\nError: S3 credentials not found", file=sys.stderr)
            print("Copy .env.example to .env and set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.", file=sys.stderr)
            sys.exit(1)

        # Create S3 client
        import boto3

        s3_client = boto3.client("s3", **s3_config)

        # Ensure bucket exists
        region = s3_config.get("region_name", "us-east-1")
        try:
            ensure_s3_bucket_exists(s3_client, upload_bucket, region=region)
        except Exception as e:
            print(f"\nError ensuring S3 bucket exists: {e}", file=sys.stderr)
            sys.exit(1)

        # Determine S3 prefix with proper granulation including format
        if args.s3_prefix is None:
            if args.dataset == "beir":
                # Structure: datasets/rag/beir/{beir_dataset}/{format}/{num_samples}
                args.s3_prefix = f"datasets/rag/beir/{args.beir_dataset}/{args.output_format}/{args.num_samples}"
            elif args.dataset == "open_ragbench":
                # Structure: datasets/rag/open_ragbench/arxiv/{format}/{num_samples}
                args.s3_prefix = f"datasets/rag/open_ragbench/arxiv/{args.output_format}/{args.num_samples}"
            elif args.dataset == "slidevqa":
                # Structure: datasets/rag/slidevqa/{split}/{format}/{num_samples}
                args.s3_prefix = f"datasets/rag/slidevqa/{args.slidevqa_split}/{args.output_format}/{args.num_samples}"
            elif args.dataset == "html_rag":
                # Structure: datasets/rag/html_rag/{split}/html/{num_samples}
                args.s3_prefix = f"datasets/rag/html_rag/{args.nq_split}/html/{args.num_samples}"
            elif args.dataset == "nomiracl":
                # Structure: datasets/rag/nomiracl/{language}/{split}/{format}/{num_samples}
                args.s3_prefix = f"datasets/rag/nomiracl/{args.nomiracl_language}/{args.nomiracl_split}/{args.output_format}/{args.num_samples}"
            elif args.dataset == "mlqa":
                # Structure: datasets/rag/mlqa/{language}/{split}/{format}/{num_samples}
                args.s3_prefix = f"datasets/rag/mlqa/{args.mlqa_language}/{args.mlqa_split}/{args.output_format}/{args.num_samples}"
            elif args.dataset == "mkqa":
                # Structure: datasets/rag/mkqa/{language}/{format}/{num_samples}
                args.s3_prefix = f"datasets/rag/mkqa/{args.mkqa_language}/{args.output_format}/{args.num_samples}"
            else:
                # Fallback for other datasets
                args.s3_prefix = f"datasets/rag/{args.dataset}/{args.output_format}/{args.num_samples}"

        # Upload
        print(f"\n{'='*60}")
        print(f"Uploading to S3")
        print(f"{'='*60}")
        print(f"Bucket: {upload_bucket}")
        print(f"Prefix: {args.s3_prefix}")
        print()

        try:
            input_data_key, test_data_key = upload_dataset_to_s3(
                s3_client,
                local_kb_dir=kb_dir,
                local_bench_path=bench_path,
                bucket=upload_bucket,
                prefix=args.s3_prefix,
            )
        except Exception as e:
            print(f"\nError uploading to S3: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Print manifest YAML snippet
    print(f"\n{'='*60}")
    print(f"Add to dataset_manifest.yaml:")
    print(f"{'='*60}")

    # Generate dataset ID
    dataset_id = args.dataset
    dataset_name = args.dataset.replace("_", " ").title()

    if args.dataset == "beir":
        dataset_id = f"beir-{args.beir_dataset}-{args.num_samples}"
        dataset_name = f"BEIR {args.beir_dataset.title()} ({args.num_samples} samples)"
    elif args.dataset == "open_ragbench":
        dataset_id = f"open-ragbench-arxiv-{args.num_samples}"
        dataset_name = f"Open RAGBench ArXiv ({args.num_samples} samples)"
    elif args.dataset == "slidevqa":
        dataset_id = f"slidevqa-{args.slidevqa_split}-{args.num_samples}"
        dataset_name = f"SlideVQA {args.slidevqa_split.title()} ({args.num_samples} samples)"
    elif args.dataset == "html_rag":
        dataset_id = f"html-rag-nq-{args.nq_split}-{args.num_samples}"
        dataset_name = f"HtmlRag NQ {args.nq_split.title()} ({args.num_samples} samples)"
    elif args.dataset == "nomiracl":
        dataset_id = f"nomiracl-{args.nomiracl_language}-{args.nomiracl_split}-{args.num_samples}-{args.output_format}"
        dataset_name = f"NoMIRACL {args.nomiracl_language.upper()} {args.nomiracl_split.title()} ({args.num_samples}) {args.output_format.upper()}"
    elif args.dataset == "mlqa":
        dataset_id = f"mlqa-{args.mlqa_language}-{args.mlqa_split}-{args.num_samples}-{args.output_format}"
        dataset_name = f"MLQA {args.mlqa_language.upper()} {args.mlqa_split.title()} ({args.num_samples}) {args.output_format.upper()}"
    elif args.dataset == "mkqa":
        dataset_id = f"mkqa-{args.mkqa_language}-{args.num_samples}-{args.output_format}"
        dataset_name = f"MKQA {args.mkqa_language.upper()} ({args.num_samples}) {args.output_format.upper()}"

    if args.upload_to_s3:
        print(f"""
- id: {dataset_id}
  name: "{dataset_name}"
  input_data_key: "{input_data_key}"
  test_data_key: "{test_data_key}"
  optimization_metric: "faithfulness"
  embeddings_models:
    - "vllm-embedding/bge-m3"
""")
    else:
        print(f"""
# Dataset generated locally at: {args.output_dir}
# Upload to S3 first using --upload-to-s3 flag, then add entry like:
#
# - id: {dataset_id}
#   name: "{dataset_name}"
#   input_data_key: "your-s3-prefix/knowledge_base"
#   test_data_key: "your-s3-prefix/benchmark_data.json"
#   optimization_metric: "faithfulness"
#   embeddings_models:
#     - "vllm-embedding/bge-m3"
""")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
