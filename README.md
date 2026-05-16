---
title: GO Emotion Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
pinned: false
license: mit
---

# 🧠 GO Emotion Classifier — Multi-Label Emotion Detection

<div align="center">

**Transformer-based emotion intelligence app that detects dominant and secondary emotions from text with confidence scores, visual breakdowns, and supportive next-step suggestions.**

[![CI/CD](https://github.com/JEEVAN-DR/psysense-emotion-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/JEEVAN-DR/psysense-emotion-ai/actions)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-blue?style=for-the-badge&logo=streamlit)](https://psysense-emotion-ai-g8ugdajqqquupqjjpxqfiz.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/JEEVAN-DR/psysense-emotion-ai)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-DistilBERT-orange?style=for-the-badge)](https://huggingface.co/)

*Enter text, analyze emotional signals, and view multi-label emotion probabilities powered by DistilBERT and the GoEmotions taxonomy.*

</div>

---

## Overview

This project is a Transformer-based Natural Language Processing system that detects multiple simultaneous human emotions from text.

Unlike traditional sentiment analysis models that predict a single label such as positive or negative, this application performs multi-label emotion classification. It identifies dominant and secondary emotions with confidence scores.

The model is based on DistilBERT and the GoEmotions emotion taxonomy, with support for threshold-controlled multi-label predictions and interactive visual analysis.

Built by **Jeevan D R**, a **Top 1% Student** focused on applied AI, NLP, and production-ready machine learning tools.

## Key Features

* Multi-label emotion detection across 28 emotions
* Dominant emotion identification
* Secondary emotion detection
* Confidence-based ranking of emotions
* Threshold-controlled emotion activation
* Real-time prediction using Streamlit
* Transformer-based inference with HuggingFace Transformers
* Interactive emotion probability visualization
* FastAPI service for production-style inference
* Docker, Prometheus, Grafana, and MLflow support

## Model Details

* Base Model: DistilBERT uncased
* Task: Multi-label emotion classification
* Dataset: GoEmotions
* Number of Labels: 28 emotions
* Loss Function: Binary Cross-Entropy with logits
* Max Sequence Length: 128 tokens

## Example Prediction

Input:

```text
I love you but I am scared
```

Output:

```text
Dominant Emotion: love

Other Detected Emotions:
- fear
- nervousness

Emotion Probability Distribution shown as a bar chart.
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/JEEVAN-DR/psysense-emotion-ai.git
cd psysense-emotion-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Model Weights

Model weights are loaded from HuggingFace Hub during inference. Set `HF_MODEL_ID` to the HuggingFace model repository that contains the trained DistilBERT weights.

Example:

```bash
set HF_MODEL_ID=JEEVAN-DR/psysense-emotion-ai
```

The default model ID in the code is:

```text
Hitan2004/psysense-emotion-ai
```

This fallback keeps the Streamlit app running until Jeevan uploads his own trained model weights. To make the deployment fully use Jeevan's HuggingFace model, create `JEEVAN-DR/psysense-emotion-ai` on HuggingFace and set `HF_MODEL_ID` to that repository in Streamlit Cloud.

## Project Structure

```text
psysense-emotion-ai/
|-- streamlit_app.py        # Streamlit UI
|-- inference.py            # Prediction pipeline
|-- api/                    # FastAPI service
|-- model/
|   `-- label_encoder.pkl
|-- monitoring/             # Prometheus configuration
|-- k8s/                    # Kubernetes manifests
|-- requirements.txt
|-- requirements-prod.txt
|-- Dockerfile
`-- README.md
```

## Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* Streamlit
* FastAPI
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Docker

## Use Cases

* Emotion-aware conversational AI
* Mental health text analysis tools
* Human-computer interaction systems
* Social media emotion analytics
* Customer feedback emotion detection
* Affective computing research

## Author

**Jeevan D R**  
Top 1% Student  
AI & NLP Developer - Emotion Recognition  
Mysuru, India

## License

MIT License.
