# Red Hat OpenShift AI Overview

Red Hat OpenShift AI is a flexible and scalable AI and machine learning platform built on Red Hat OpenShift. It enables enterprises to create, deploy, and manage AI-enabled applications at scale across hybrid cloud environments.

## Key Features

- **Model Training**: Distributed training support for large language models using PyTorch and TensorFlow.
- **Model Serving**: Production-grade model serving with vLLM and Red Hat Inference Server.
- **AutoRAG**: Automated Retrieval-Augmented Generation pipeline optimization for enterprise document corpora.
- **Data Science Pipelines**: Kubeflow-based pipelines for reproducible ML workflows.

## AutoRAG Pipeline

The AutoRAG pipeline automates the optimization of RAG configurations. It discovers documents from S3 storage, extracts text using Docling, and runs hyperparameter optimization to find the best chunking, embedding, and retrieval settings for a given corpus.

### Supported Document Formats

AutoRAG supports a wide range of document formats including PDF, DOCX, PPTX, Markdown, HTML, plain text, OpenDocument formats (ODT, ODP), AsciiDoc, LaTeX, EPUB, and email formats (EML). This broad format support ensures maximum corpus coverage for enterprise document collections.