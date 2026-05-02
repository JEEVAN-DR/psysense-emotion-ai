# 🧠 GoEmotions Emotion Classifier — DistilBERT Multi-Label NLP

🔗 **Live Demo:**
https://psysense-emotion-ai-75rqc6axets9shtwledjui.streamlit.app/

---

## 📌 Overview

**PsySense** is a Transformer-based Natural Language Processing system that detects **multiple simultaneous human emotions from text**.

Unlike traditional sentiment analysis models that predict a single label (positive/negative), PsySense performs **multi-label emotion classification**, identifying both **dominant and secondary emotions with confidence scores**.

The model is fine-tuned on Google’s **GoEmotions dataset** using **DistilBERT** with **class-weighted Binary Cross-Entropy Loss** to handle class imbalance and real-world emotional overlap.

---

## 🚀 Key Features

* Multi-label emotion detection (28 emotions)
* Dominant emotion identification
* Secondary emotion detection
* Confidence-based ranking of emotions
* Threshold-controlled emotion activation
* Real-time prediction using Streamlit UI
* Transformer fine-tuning with HuggingFace
* Class imbalance handling using weighted BCE loss
* Interactive emotion probability visualization

---

## 🧠 Model Details

* **Base Model:** DistilBERT (uncased)
* **Task:** Multi-Label Emotion Classification
* **Dataset:** GoEmotions (Google Research)
* **Number of Labels:** 28 emotions
* **Loss Function:** BCEWithLogitsLoss (with class weights)
* **Evaluation Metrics:** Micro-F1, Macro-F1
* **Max Sequence Length:** 128 tokens

---

## 📊 Example Prediction

**Input**

```
I love you but I am scared
```

**Output**

```
Dominant Emotion: Love (82%)

Other Detected Emotions:
• Fear — 67%  
• Nervousness — 41%  

Emotion Probability Distribution shown as bar graph.
```

---

## 🌐 Live Deployment

The model is deployed as an interactive web application using **Streamlit Cloud**.

Users can enter text and instantly receive:

* Dominant emotion
* Secondary emotions
* Confidence scores
* Emotion probability graph

---

## ⚙️ Run Locally

Clone the repository:

```
git clone https://github.com/Hitan2004/psysense-emotion-ai.git
cd psysense-emotion-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

Run Streamlit app:

```
streamlit run streamlit_app.py
```

---

## 🤖 Model Weights

Model weights are hosted on **HuggingFace Hub** and will be automatically downloaded during inference.

HuggingFace Model:
`Hitan2004/psysense-emotion-ai`

---

## 📂 Project Structure

```
psysense-emotion-ai/
│
├── streamlit_app.py        # Streamlit UI  
├── inference.py            # Prediction pipeline  
├── model/
│   └── label_encoder.pkl  
├── requirements.txt  
├── README.md  
└── .gitignore  
```

---

## 🛠️ Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

## 🎯 Use Cases

* Emotion-aware conversational AI
* Mental health text analysis tools
* Human-computer interaction systems
* Social media emotion analytics
* Customer feedback emotion detection
* Affective computing research

---

## 👨‍💻 Author

**Hitan K**
AI/ML Engineer — NLP & Emotion Recognition
Bengaluru, India

---

⭐ If you found this project interesting, consider giving it a star.
