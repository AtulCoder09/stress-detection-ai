"""
╔══════════════════════════════════════════════════════════════╗
║        🧠 AI Stress Detection — Face + Voice Fusion         ║
║     EfficientNet-B2 (Face) + Wav2Vec2 (Voice) Combined      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import librosa
from PIL import Image
from torchvision import transforms, models
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ═══════════════════════════════════════════════════════════
#  DEVICE
# ═══════════════════════════════════════════════════════════
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"🖥️  Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════
#  FACE MODEL — EfficientNet-B2 + CBAM  (7 classes)
# ═══════════════════════════════════════════════════════════
FACE_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_FACE_CLASSES = len(FACE_CLASSES)


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(),
            nn.Linear(ch // r, ch, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        return torch.sigmoid(self.fc(avg) + self.fc(mx)).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class EmotionModel(nn.Module):
    """Face emotion model — EfficientNet-B2 backbone + CBAM attention.
    
    Loads efficientnet_b2.pth as the backbone pretrained weights,
    identical to the training notebook setup.
    """

    def __init__(self, num_classes, backbone_weights_path="efficientnet_b2.pth"):
        super().__init__()
        backbone = models.efficientnet_b2(weights=None)

        # Load pretrained EfficientNet-B2 weights from local file (matching notebook)
        if os.path.exists(backbone_weights_path):
            state_dict = torch.load(backbone_weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict)
            print(f"   Loaded backbone weights from {backbone_weights_path}")
        else:
            print(f"   WARNING: {backbone_weights_path} not found — backbone using random init")

        self.backbone = backbone.features
        self.cbam = CBAM(1408)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1408 * 2),
            nn.Linear(1408 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)
        x = torch.cat([avg, mx], dim=1)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════════════════
#  VOICE MODEL — Wav2Vec2 + Classifier
# ═══════════════════════════════════════════════════════════
class Wav2Vec2Classifier(nn.Module):
    """Voice emotion model — Wav2Vec2 frozen backbone + classification head."""

    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base", use_safetensors=False
        )
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)

class AdvancedWav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base", use_safetensors=False
        )
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        for layer in self.wav2vec2.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)


# ═══════════════════════════════════════════════════════════
#  STRESS MAPPING — Per-Emotion Stress Weights
# ═══════════════════════════════════════════════════════════
#  Each emotion contributes a different amount to the overall
#  "stress" score. Negative emotions → high stress contribution.
#  This is a psychologically-informed mapping.

STRESS_WEIGHTS = {
    # High-stress emotions
    "angry":     0.90,
    "anger":     0.90,
    "disgust":   0.70,
    "fear":      0.85,
    "fearful":   0.85,
    "sad":       0.75,
    "sadness":   0.75,
    # Low-stress emotions
    "happy":     0.10,
    "happiness": 0.10,
    "neutral":   0.25,
    "calm":      0.15,
    "surprise":  0.50,
    "surprised": 0.50,
    "ps":        0.50,  # pleasant surprise
    # New V2 Stress Intensity Classes
    "calm":      0.10,
    "low":       0.30,
    "moderate":  0.60,
    "high":      0.85,
    "extreme":   0.98,
}


def get_stress_weight(class_name: str) -> float:
    """Return the stress weight for a given emotion class."""
    return STRESS_WEIGHTS.get(class_name.lower().strip(), 0.50)


# ═══════════════════════════════════════════════════════════
#  LOAD MODELS
# ═══════════════════════════════════════════════════════════
print("📦 Loading Face Model …")
FACE_V2_PATH = os.path.join("models", "face_model_v2.pth")
VOICE_V2_PATH = os.path.join("models", "voice_model_v2.pth")

face_model = EmotionModel(NUM_FACE_CLASSES).to(DEVICE)
face_ckpt_path = FACE_V2_PATH if os.path.exists(FACE_V2_PATH) else "face_model.pth"
print(f"   Using weights: {face_ckpt_path}")

face_ckpt = torch.load(face_ckpt_path, map_location=DEVICE, weights_only=False)
if isinstance(face_ckpt, dict):
    if "model_state_dict" in face_ckpt:
        face_model.load_state_dict(face_ckpt["model_state_dict"])
    elif "model" in face_ckpt:
        face_model.load_state_dict(face_ckpt["model"])
    else:
        face_model.load_state_dict(face_ckpt)
else:
    face_model.load_state_dict(face_ckpt)
face_model.eval()
print(f"   ✅ Face Model loaded  |  Classes: {FACE_CLASSES}")

print("📦 Loading Voice Model …")
is_v2_voice = os.path.exists(VOICE_V2_PATH)
voice_ckpt_path = VOICE_V2_PATH if is_v2_voice else "super_fast_model.pth"
print(f"   Using weights: {voice_ckpt_path}")

voice_ckpt = torch.load(voice_ckpt_path, map_location=DEVICE, weights_only=False)
voice_label_encoder = voice_ckpt["label_encoder"]
VOICE_CLASSES = list(voice_label_encoder.classes_)
NUM_VOICE_CLASSES = len(VOICE_CLASSES)

if is_v2_voice:
    voice_model = AdvancedWav2Vec2Classifier(num_classes=NUM_VOICE_CLASSES, dropout=0.5).to(DEVICE)
else:
    voice_model = Wav2Vec2Classifier(num_classes=NUM_VOICE_CLASSES, dropout=0.5).to(DEVICE)

voice_model.load_state_dict(voice_ckpt["model_state_dict"])
voice_model.eval()
print(f"   ✅ Voice Model loaded |  Classes: {VOICE_CLASSES}")

print("📦 Loading Wav2Vec2 Processor …")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
print("   ✅ Processor ready")

# ═══════════════════════════════════════════════════════════
#  PREPROCESSING PIPELINES
# ═══════════════════════════════════════════════════════════
face_transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

SAMPLE_RATE = 16_000
MAX_DURATION = 3.0


# ═══════════════════════════════════════════════════════════
#  INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════
def predict_face(image: Image.Image):
    """Run face emotion prediction, returns softmax probabilities."""
    img = image.convert("RGB")
    tensor = face_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = face_model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return dict(zip(FACE_CLASSES, probs.tolist()))


def predict_voice(audio_path: str):
    """Run voice emotion prediction, returns softmax probabilities."""
    max_len = int(SAMPLE_RATE * MAX_DURATION)
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    inputs = processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = voice_model(inputs.input_values.to(DEVICE))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return dict(zip(VOICE_CLASSES, probs.tolist()))


# ═══════════════════════════════════════════════════════════
#  FUSION LOGIC
# ═══════════════════════════════════════════════════════════
FACE_WEIGHT = 0.6
VOICE_WEIGHT = 0.4


def compute_stress(face_probs: dict | None, voice_probs: dict | None):
    """
    Compute final stress percentage by fusing face & voice predictions.

    Strategy:
      1. For each model, compute a stress score as:
            stress_score = Σ  P(emotion_i) × stress_weight(emotion_i)
         This gives a value in [0, 1] representing how "stressed"
         the detected emotional state is.

      2. Combine:
            final = 0.6 × face_stress + 0.4 × voice_stress

      3. If only one modality is available, use that alone (weight = 1.0).
    """
    face_stress = None
    voice_stress = None

    if face_probs:
        face_stress = sum(
            prob * get_stress_weight(cls) for cls, prob in face_probs.items()
        )

    if voice_probs:
        voice_stress = sum(
            prob * get_stress_weight(cls) for cls, prob in voice_probs.items()
        )

    # Determine active weights
    if face_stress is not None and voice_stress is not None:
        final = FACE_WEIGHT * face_stress + VOICE_WEIGHT * voice_stress
    elif face_stress is not None:
        final = face_stress
    elif voice_stress is not None:
        final = voice_stress
    else:
        final = 0.0

    return round(final * 100, 1), face_stress, voice_stress


# ═══════════════════════════════════════════════════════════
#  MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════
def analyze(image, audio):
    """
    Main function called by Gradio.
    Accepts an image (PIL/ndarray) and audio (filepath),
    returns stress %, face breakdown, voice breakdown, and summary.
    """
    face_probs = None
    voice_probs = None

    # ── Face inference ──
    if image is not None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        face_probs = predict_face(image)

    # ── Voice inference ──
    audio_path = None
    if audio is not None:
        audio_path = audio if isinstance(audio, str) else audio
        voice_probs = predict_voice(audio_path)

    # ── Fusion ──
    stress_pct, face_s, voice_s = compute_stress(face_probs, voice_probs)

    # ── Build detailed results ──
    face_result = {}
    voice_result = {}

    if face_probs:
        face_result = {cls.capitalize(): round(p * 100, 1) for cls, p in face_probs.items()}
    if voice_probs:
        voice_result = {cls.capitalize(): round(p * 100, 1) for cls, p in voice_probs.items()}

    # ── Stress level label ──
    if stress_pct >= 75:
        level = "🔴 High Stress"
        advice = "Consider taking a break, deep breathing, or talking to someone."
    elif stress_pct >= 50:
        level = "🟠 Moderate Stress"
        advice = "You seem a bit tense. A short walk or music might help."
    elif stress_pct >= 30:
        level = "🟡 Mild Stress"
        advice = "Slight signs of stress detected. Stay mindful!"
    else:
        level = "🟢 Low / No Stress"
        advice = "You seem calm and relaxed. Keep it up! 😊"

    # ── Summary HTML ──
    modalities_used = []
    if face_probs:
        modalities_used.append("Face")
    if voice_probs:
        modalities_used.append("Voice")
    modality_str = " + ".join(modalities_used) if modalities_used else "None"

    # Determine dominant emotions
    face_dominant = ""
    voice_dominant = ""
    if face_probs:
        top_face = max(face_probs, key=face_probs.get)
        face_dominant = f"{top_face.capitalize()} ({face_probs[top_face]*100:.1f}%)"
    if voice_probs:
        top_voice = max(voice_probs, key=voice_probs.get)
        voice_dominant = f"{top_voice.capitalize()} ({voice_probs[top_voice]*100:.1f}%)"

    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 20px;
        padding: 30px;
        color: #fff;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    ">
        <!-- Stress Gauge -->
        <div style="text-align:center; margin-bottom:25px;">
            <div style="
                font-size: 4rem;
                font-weight: 800;
                background: linear-gradient(135deg,
                    {'#00ff87, #60efff' if stress_pct < 30 else
                     '#ffd700, #ff9500' if stress_pct < 50 else
                     '#ff6b6b, #ee5a24' if stress_pct < 75 else
                     '#ff0000, #c0392b'});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 30px rgba(255,255,255,0.1);
            ">{stress_pct}%</div>
            <div style="font-size:1.4rem; margin-top:5px; font-weight:600;">{level}</div>
        </div>

        <!-- Progress Bar -->
        <div style="
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            height: 20px;
            margin: 15px 0 25px 0;
            overflow: hidden;
            backdrop-filter: blur(10px);
        ">
            <div style="
                width: {stress_pct}%;
                height: 100%;
                border-radius: 12px;
                background: linear-gradient(90deg,
                    {'#00ff87, #60efff' if stress_pct < 30 else
                     '#ffd700, #ff9500' if stress_pct < 50 else
                     '#ff6b6b, #ee5a24' if stress_pct < 75 else
                     '#ff0000, #c0392b'});
                transition: width 1s ease-in-out;
                box-shadow: 0 0 15px rgba(255,255,255,0.3);
            "></div>
        </div>

        <!-- Details Grid -->
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        ">
            <div style="
                background: rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 15px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size:0.85rem; color:#aaa; margin-bottom:5px;">📸 Face Emotion</div>
                <div style="font-size:1.1rem; font-weight:600;">
                    {face_dominant if face_dominant else '—  No image provided'}
                </div>
                <div style="font-size:0.8rem; color:#888; margin-top:4px;">
                    Weight: {FACE_WEIGHT*100:.0f}%
                    {f' | Stress Score: {face_s*100:.1f}%' if face_s is not None else ''}
                </div>
            </div>
            <div style="
                background: rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 15px;
                backdrop-filter: blur(5px);
            ">
                <div style="font-size:0.85rem; color:#aaa; margin-bottom:5px;">🎙️ Voice Emotion</div>
                <div style="font-size:1.1rem; font-weight:600;">
                    {voice_dominant if voice_dominant else '—  No audio provided'}
                </div>
                <div style="font-size:0.8rem; color:#888; margin-top:4px;">
                    Weight: {VOICE_WEIGHT*100:.0f}%
                    {f' | Stress Score: {voice_s*100:.1f}%' if voice_s is not None else ''}
                </div>
            </div>
        </div>

        <!-- Advice -->
        <div style="
            background: rgba(255,255,255,0.05);
            border-left: 3px solid {'#00ff87' if stress_pct < 30 else '#ffd700' if stress_pct < 50 else '#ff6b6b' if stress_pct < 75 else '#ff0000'};
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 10px;
        ">
            <div style="font-size:0.85rem; color:#aaa;">💡 Suggestion</div>
            <div style="font-size:0.95rem; margin-top:4px;">{advice}</div>
        </div>

        <!-- Footer -->
        <div style="
            text-align: center;
            margin-top: 18px;
            font-size: 0.75rem;
            color: #666;
        ">
            Modalities: {modality_str} &nbsp;|&nbsp;
            Face × {FACE_WEIGHT} + Voice × {VOICE_WEIGHT}
        </div>
    </div>
    """

    return summary_html, face_result, voice_result


# ═══════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════
TITLE = "🧠 AI Stress Detection"
DESCRIPTION = """
<div style="text-align:center; max-width:700px; margin:auto;">
    <p style="font-size:1.1rem; color:#ccc;">
        Upload a <strong>face image</strong> and/or <strong>voice recording</strong>
        to detect your stress level using deep learning.
    </p>
    <p style="font-size:0.9rem; color:#888;">
        <strong>Face Model</strong>: EfficientNet-B2 + CBAM Attention (weight 0.6)<br>
        <strong>Voice Model</strong>: Wav2Vec2 Frozen + Classifier (weight 0.4)<br>
        Both models predict emotion probabilities, which are then mapped
        to a unified <strong>stress percentage</strong>.
    </p>
</div>
"""

css = """
/* ── Global ── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 1100px !important;
    margin: auto !important;
}

/* ── Title ── */
h1 {
    text-align: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* ── Labels ── */
.label-wrap {
    font-weight: 600 !important;
}

/* ── Submit Button ── */
#submit-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 40px !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
    transition: all 0.3s ease !important;
}

#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5) !important;
}

/* ── Footer ── */
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="0px",
    block_shadow="0 4px 20px rgba(0,0,0,0.3)",
    input_background_fill="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg, #667eea, #764ba2)",
    button_primary_text_color="white",
)

with gr.Blocks(theme=theme, css=css, title="AI Stress Detection") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.HTML(DESCRIPTION)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="📸 Upload Face Image",
                type="pil",
                sources=["upload", "webcam"],
                height=300,
            )
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎙️ Upload Voice Recording",
                type="filepath",
                sources=["upload", "microphone"],
            )

    submit_btn = gr.Button(
        "🔍  Analyze Stress Level",
        variant="primary",
        elem_id="submit-btn",
    )

    gr.Markdown("---")
    gr.Markdown("### 📊 Results")

    result_html = gr.HTML(label="Stress Analysis")

    with gr.Row():
        face_output = gr.Label(label="📸 Face Emotion Breakdown (%)", num_top_classes=7)
        voice_output = gr.Label(label="🎙️ Voice Emotion Breakdown (%)", num_top_classes=7)

    submit_btn.click(
        fn=analyze,
        inputs=[img_input, audio_input],
        outputs=[result_html, face_output, voice_output],
    )

    gr.Markdown(
        """
        <div style="text-align:center; padding:20px; color:#555; font-size:0.8rem;">
            Built with ❤️ using PyTorch, Wav2Vec2, EfficientNet-B2 & Gradio<br>
            Face Model Weight: 0.6 &nbsp;|&nbsp; Voice Model Weight: 0.4
        </div>
        """
    )

# ═══════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch()
