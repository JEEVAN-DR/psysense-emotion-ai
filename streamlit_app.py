import streamlit as st

# ── MUST be the very first Streamlit call ─────────────────────
st.set_page_config(
    page_title="GO Emotion Classifier",
    page_icon="🎭",
    layout="wide"
)

# ── Safe to import now ────────────────────────────────────────
from inference import (
    load_model, predict_emotions, plot_emotions,
    explain_emotion, get_emoji, give_advice
)


# ── Load model once, cache forever ───────────────────────────
@st.cache_resource(show_spinner="Loading emotion model...")
def get_model():
    return load_model()

model, tokenizer, mlb, device = get_model()
label_names = list(mlb.classes_)


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
# 🎭 GO Emotion Classifier
### Detect and understand human emotions from text
""")
st.divider()


# ── Input ─────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
        "✍️ Share what's on your mind...",
        placeholder="Example: I feel proud but also nervous about tomorrow...",
        height=180
    )
    analyze = st.button("🔍 Analyze My Emotions", use_container_width=True)

with col2:
    st.info("""
### 💡 How it works
- Fine-tuned DistilBERT transformer
- Detects up to 28 distinct emotions
- Multi-label: multiple emotions at once
- Confidence scoring per emotion
- Personalized emotional advice
""")

st.divider()


# ── Results ───────────────────────────────────────────────────
if analyze:
    if not text or not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing your emotions..."):
            result = predict_emotions(
                model, tokenizer, label_names, device, text,
                threshold=0.10,   # consistent threshold
                top_k=10
            )

        if "error" in result:
            st.error(result["error"])

        else:
            emotion    = result["dominant_emotion"]["label"]
            confidence = result["dominant_emotion"]["confidence"]
            emoji      = get_emoji(emotion)

            # Secondary = active emotions excluding dominant
            # FIX: use top_emotions with 0.08 display floor so
            #      near-threshold emotions are never silently hidden
            secondary = [
                e for e in result["active_emotions"]
                if e["label"] != emotion
            ]

            # Fallback: if active_emotions missed anything above 8%,
            # pull from top_emotions so the UI never shows empty
            if not secondary:
                secondary = [
                    {"label": lbl, "confidence": prob}
                    for lbl, prob in result["top_emotions"]
                    if lbl != emotion and prob >= 0.08
                ]

            # ── AI Insight ────────────────────────────────────
            st.markdown("## 🤖 AI Emotional Insight")

            if confidence < 0.30:
                # Mixed / uncertain
                blend = ", ".join(
                    f"**{e['label'].capitalize()}** ({e['confidence']*100:.0f}%)"
                    for e in secondary[:3]
                )
                msg = (
                    f"💭 **Your emotions seem mixed.**\n\n"
                    f"Strongest signal: **{emotion.capitalize()}** "
                    f"({confidence*100:.0f}%)"
                )
                if blend:
                    msg += f", alongside {blend}."
                st.warning(msg)

            elif secondary:
                blend = " and ".join(
                    f"**{e['label'].capitalize()}**" for e in secondary[:3]
                )
                st.success(
                    f"{emoji} **You seem to be feeling "
                    f"{emotion.capitalize()} alongside {blend}.**\n\n"
                    f"{explain_emotion(emotion)}"
                )

            else:
                st.success(
                    f"{emoji} **You seem to be feeling "
                    f"{emotion.capitalize()}.**\n\n"
                    f"{explain_emotion(emotion)}"
                )

            # Advice — cover all detected emotions
            emotions_to_advise = [emotion] + [e["label"] for e in secondary[:2]]
            if len(emotions_to_advise) > 1:
                parts = [
                    f"**For your {em.capitalize()}:** {give_advice(em)}"
                    for em in emotions_to_advise
                ]
                st.info("### 🌱 Suggested Next Steps\n\n" + "\n\n".join(parts))
            else:
                st.info(f"### 🌱 Suggested Next Step\n{give_advice(emotion)}")

            st.divider()

            # ── Dominant emotion card ─────────────────────────
            st.markdown("## 🎯 Dominant Emotion")

            if confidence < 0.30:
                st.caption("⚠️ Low confidence — your emotions appear mixed")

            m1, m2, m3 = st.columns(3)
            m1.metric("Emotion",    emotion.capitalize())
            m2.metric("Confidence", f"{confidence*100:.1f}%")
            m3.metric("Emoji",      emoji)
            st.progress(min(confidence, 1.0))
            st.divider()

            # ── Secondary emotions ────────────────────────────
            st.markdown("## 🔁 Other Detected Emotions")

            if secondary:
                for e in secondary:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    c1.write(f"{get_emoji(e['label'])}")
                    c2.progress(min(e["confidence"], 1.0),
                                text=e["label"].capitalize())
                    c3.write(f"{e['confidence']*100:.1f}%")
            else:
                st.caption("No secondary emotions detected.")

            st.divider()

            # ── Probability chart ─────────────────────────────
            st.markdown("## 📊 Emotion Probability Distribution")
            st.pyplot(plot_emotions(result))
            st.divider()

            # ── Full breakdown ────────────────────────────────
            with st.expander("🔬 Full Probability Breakdown"):
                for label, prob in result["top_emotions"]:
                    if prob > 0.005:
                        c1, c2, c3 = st.columns([1, 4, 1])
                        c1.write(get_emoji(label))
                        c2.progress(min(prob, 1.0),
                                    text=label.capitalize())
                        c3.write(f"{prob*100:.1f}%")


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
---
**GO Emotion Classifier** • Multi-label Emotion Detection  
Built with DistilBERT Transformers & Streamlit
""")
