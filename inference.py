import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)

# ── Paths ─────────────────────────────────────────────────────
HF_MODEL        = "Hitan2004/psysense-emotion-ai"
BASE_DIR        = os.path.dirname(__file__)
LOCAL_LABEL_ENC = os.path.join(BASE_DIR, "model", "label_encoder.pkl")


# ── Model loader (called once from app.py via st.cache_resource) ──
def load_model():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = DistilBertForSequenceClassification.from_pretrained(HF_MODEL)
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL)
    model.to(device)
    model.eval()
    with open(LOCAL_LABEL_ENC, "rb") as f:
        mlb = pickle.load(f)
    print("✅ Model + tokenizer + label encoder loaded")
    return model, tokenizer, mlb, device


# ── Prediction ────────────────────────────────────────────────
def predict_emotions(model, tokenizer, label_names, device,
                     text, threshold=0.10, top_k=10):
    """
    threshold=0.10  — lower than before so mixed emotions surface correctly
    top_k=10        — show more emotions in breakdown
    """
    if not text or not text.strip():
        return {"error": "Empty input text"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.sigmoid(logits)[0].cpu().numpy()
    sorted_idx = probs.argsort()[::-1]

    dominant_emotion = {
        "label":      label_names[sorted_idx[0]],
        "confidence": float(probs[sorted_idx[0]])
    }

    # active = everything above threshold
    active_emotions = [
        {"label": label_names[i], "confidence": float(probs[i])}
        for i in sorted_idx
        if probs[i] >= threshold
    ]

    # top_k for chart and breakdown
    top_emotions = [
        (label_names[i], float(probs[i]))
        for i in sorted_idx[:top_k]
    ]

    return {
        "dominant_emotion": dominant_emotion,
        "active_emotions":  active_emotions,
        "top_emotions":     top_emotions
    }


# ── All 28 emotion explanations ───────────────────────────────
EXPLANATIONS = {
    "admiration":    "You're expressing genuine respect or appreciation for someone.",
    "amusement":     "Something struck you as funny or entertaining.",
    "anger":         "You're experiencing strong frustration or rage.",
    "annoyance":     "Something is irritating or mildly frustrating you.",
    "approval":      "You're feeling agreement or positive acknowledgment.",
    "caring":        "You're showing concern, kindness, or emotional support.",
    "confusion":     "You're feeling uncertain or finding something hard to understand.",
    "curiosity":     "You're interested in learning or discovering something new.",
    "desire":        "You're strongly wanting or longing for something.",
    "disappointment":"Your expectations weren't met — that's a valid feeling.",
    "disapproval":   "You're expressing disagreement or criticism about something.",
    "disgust":       "Something is triggering strong dislike or revulsion in you.",
    "embarrassment": "You're feeling awkward or ashamed about something.",
    "excitement":    "You're filled with enthusiasm or positive anticipation.",
    "fear":          "You're experiencing anxiety, worry, or dread.",
    "gratitude":     "You're feeling thankful and appreciative.",
    "grief":         "You're experiencing deep sorrow or emotional loss.",
    "joy":           "You're feeling happiness and positive energy.",
    "love":          "You're experiencing deep affection or emotional attachment.",
    "nervousness":   "You're feeling tension or nervous anticipation.",
    "optimism":      "You're hopeful and expecting positive outcomes.",
    "pride":         "You're feeling accomplished or satisfied with yourself.",
    "realization":   "You just had a moment of sudden understanding or insight.",
    "relief":        "A burden has lifted — you're feeling comfort after stress.",
    "remorse":       "You're feeling guilt or regret about something.",
    "sadness":       "You're experiencing emotional pain or sorrow.",
    "surprise":      "Something unexpected caught you off guard.",
    "neutral":       "Your text doesn't carry strong emotional charge."
}


# ── All 28 advice entries ─────────────────────────────────────
ADVICE = {
    "sadness":       "You don't have to carry this alone. Reach out to someone you trust, or try journaling.",
    "grief":         "Grief takes time. Be gentle with yourself and allow the process to unfold.",
    "fear":          "Try box breathing: inhale 4s, hold 4s, exhale 4s. Writing down fears reduces their power.",
    "nervousness":   "Focus on what you can control. Preparation and grounding techniques help.",
    "anger":         "Pause before reacting. Physical movement or journaling can release tension.",
    "annoyance":     "A short break or change of environment can reset your perspective.",
    "disappointment":"Reflect on what you learned. Every unmet expectation refines your approach.",
    "disapproval":   "It's okay to disagree. Consider how to express it constructively.",
    "disgust":       "Distance yourself from what triggered this. Trust your instincts.",
    "remorse":       "Acknowledging a mistake takes courage. Think about how to make amends and move forward.",
    "embarrassment": "Everyone has awkward moments — they feel bigger to you than they look to others.",
    "confusion":     "Break the problem into smaller pieces. Asking for clarification is always a strength.",
    "joy":           "Wonderful! Share this energy or use it to work on something meaningful.",
    "excitement":    "Channel this into action — you're in a great state to start something new.",
    "love":          "Connection is one of life's greatest gifts. Nurture your relationships.",
    "gratitude":     "Consider telling the person you're grateful for how you feel.",
    "admiration":    "Let them know — expressing admiration can strengthen bonds.",
    "pride":         "Take a moment to acknowledge your effort before moving to the next goal.",
    "optimism":      "Use this mindset to tackle something you've been putting off.",
    "caring":        "Your empathy is a strength. Make sure you're also caring for yourself.",
    "curiosity":     "Follow that thread! Curiosity is the engine of learning.",
    "desire":        "Clarify what you want, then take one small step toward it today.",
    "amusement":     "Laughter is medicine. Share what made you smile.",
    "surprise":      "Take a moment to process before reacting.",
    "relief":        "Take a deep breath and enjoy this lighter feeling. You made it through.",
    "realization":   "Capture this insight before it fades — write it down.",
    "neutral":       "A calm state is a great time for focused work or learning.",
    "approval":      "It feels good to affirm others. Keep spreading that positive energy.",
}


# ── Emoji map — all 28 ────────────────────────────────────────
EMOJI_MAP = {
    "admiration": "🤩", "amusement": "😂", "anger": "😠",
    "annoyance": "😤",  "approval": "👍",  "caring": "🤗",
    "confusion": "😕",  "curiosity": "🤔", "desire": "😍",
    "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
    "embarrassment": "😳", "excitement": "🎉", "fear": "😨",
    "gratitude": "🙏",  "grief": "😢",     "joy": "😊",
    "love": "❤️",       "nervousness": "😰", "optimism": "🌟",
    "pride": "🏆",      "realization": "💡", "relief": "😮‍💨",
    "remorse": "😔",    "sadness": "😢",    "surprise": "😲",
    "neutral": "😐"
}

def explain_emotion(label):
    return EXPLANATIONS.get(label, "An emotion was detected in your text.")

def get_emoji(label):
    return EMOJI_MAP.get(label, "💭")

def give_advice(emotion):
    return ADVICE.get(emotion, "Try mindfulness, rest, or talking to someone you trust.")


# ── Chart — returns fig object (not plt module) ───────────────
def plot_emotions(result, min_prob=0.02):
    labels, scores = [], []
    for label, prob in result["top_emotions"]:
        if prob >= min_prob:
            labels.append(label.capitalize())
            scores.append(prob)

    colors = [
        "#2ecc71" if s >= 0.40 else "#f39c12" if s >= 0.15 else "#5dade2"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, max(scores) * 1.25 if scores else 1)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Emotion Probability Distribution", fontsize=13, fontweight="bold")
    ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.8, label="Threshold (0.10)")

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val*100:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig
