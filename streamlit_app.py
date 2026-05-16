import streamlit as st

st.set_page_config(
    page_title="GO Emotion Classifier",
    page_icon=":brain:",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

from inference import (
    explain_emotion,
    get_emoji,
    give_advice,
    load_model,
    plot_emotions,
    predict_emotions,
)


@st.cache_resource(show_spinner="Loading emotion model...")
def get_model():
    return load_model()


model, tokenizer, mlb, device = get_model()
label_names = list(mlb.classes_)

st.markdown(
    """
    # GO Emotion Classifier
    ### Detect and understand human emotions from text
    """
)
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
        "Share what's on your mind",
        placeholder="Example: I feel proud but also nervous about tomorrow...",
        height=180,
    )
    analyze = st.button("Analyze My Emotions", use_container_width=True)

with col2:
    st.info(
        """
        ### How it works
        - Fine-tuned DistilBERT transformer
        - Detects up to 28 distinct emotions
        - Supports multiple emotions at once
        - Scores confidence per emotion
        - Suggests next steps based on detected emotions
        """
    )

st.divider()

if analyze:
    if not text or not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing your emotions..."):
            result = predict_emotions(
                model,
                tokenizer,
                label_names,
                device,
                text,
                threshold=0.10,
                top_k=10,
            )

        if "error" in result:
            st.error(result["error"])
        else:
            emotion = result["dominant_emotion"]["label"]
            confidence = result["dominant_emotion"]["confidence"]
            emoji = get_emoji(emotion)

            secondary = [
                e for e in result["active_emotions"]
                if e["label"] != emotion
            ]

            if not secondary:
                secondary = [
                    {"label": label, "confidence": prob}
                    for label, prob in result["top_emotions"]
                    if label != emotion and prob >= 0.08
                ]

            st.markdown("## AI Emotional Insight")

            if confidence < 0.30:
                blend = ", ".join(
                    f"**{e['label'].capitalize()}** ({e['confidence'] * 100:.0f}%)"
                    for e in secondary[:3]
                )
                msg = (
                    f"**Your emotions seem mixed.**\n\n"
                    f"Strongest signal: **{emotion.capitalize()}** "
                    f"({confidence * 100:.0f}%)"
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

            emotions_to_advise = [emotion] + [e["label"] for e in secondary[:2]]
            if len(emotions_to_advise) > 1:
                parts = [
                    f"**For your {em.capitalize()}:** {give_advice(em)}"
                    for em in emotions_to_advise
                ]
                st.info("### Suggested Next Steps\n\n" + "\n\n".join(parts))
            else:
                st.info(f"### Suggested Next Step\n{give_advice(emotion)}")

            st.divider()
            st.markdown("## Dominant Emotion")

            if confidence < 0.30:
                st.caption("Low confidence - your emotions appear mixed")

            m1, m2, m3 = st.columns(3)
            m1.metric("Emotion", emotion.capitalize())
            m2.metric("Confidence", f"{confidence * 100:.1f}%")
            m3.metric("Emoji", emoji)
            st.progress(min(confidence, 1.0))

            st.divider()
            st.markdown("## Other Detected Emotions")

            if secondary:
                for item in secondary:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    c1.write(get_emoji(item["label"]))
                    c2.progress(
                        min(item["confidence"], 1.0),
                        text=item["label"].capitalize(),
                    )
                    c3.write(f"{item['confidence'] * 100:.1f}%")
            else:
                st.caption("No secondary emotions detected.")

            st.divider()
            st.markdown("## Emotion Probability Distribution")
            st.pyplot(plot_emotions(result))

            st.divider()
            with st.expander("Full Probability Breakdown"):
                for label, prob in result["top_emotions"]:
                    if prob > 0.005:
                        c1, c2, c3 = st.columns([1, 4, 1])
                        c1.write(get_emoji(label))
                        c2.progress(min(prob, 1.0), text=label.capitalize())
                        c3.write(f"{prob * 100:.1f}%")

st.markdown(
    """
    ---
    **GO Emotion Classifier** - Multi-label Emotion Detection
    """
)
