# asl_webcam_infer.py

import json
import math
import time
from pathlib import Path
from collections import Counter

import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms

import pennylane as qml
from pennylane import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = (112, 112)   # must match training
H, W = IMG_SIZE

E = 128        # ResNet embedding dim (same as training)
D_EMBED = 8    # dim fed into VQC
N_QUBITS = 4
VQC_LAYERS = 2

BEST_IMG_PATH = "best_img.pth"
CONFIG_PATH   = "config.json"
OUT_JSON      = "webcam_infer_log.json"   # log file

# --- NEW: timing settings ---
COUNTDOWN_SEC    = 5.0   # seconds before recording starts
RECORD_DURATION  = 15.0  # seconds to record per segment


# ====== LOAD CONFIG / CLASSES ======

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

CLASSES = cfg["classes"]   # e.g. ["A","B","C","D","E"]
NUM_CLASSES = len(CLASSES)

print("[INFO] Loaded classes:", CLASSES)


val_trsfm_img = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


# ====== MODEL DEFINITIONS (same as training) ======

class ResNetEmbed(nn.Module):
    def __init__(self, emb_dim=E, num_classes=NUM_CLASSES,
                 pretrained=False, freeze_stem=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_stem:
            for name, p in self.backbone.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1")):
                    p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(inplace=True)
        )
        self.head_cls = nn.Linear(emb_dim, NUM_CLASSES)
        self.reduce_to_d = nn.Linear(emb_dim, D_EMBED)

    def forward(self, x, return_embed_only=False):
        features512 = self.backbone(x)
        emb = self.proj(features512)
        if return_embed_only:
            return emb
        logits = self.head_cls(emb)
        return emb, logits


class CNNtoDE(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        emb = self.cnn(x, return_embed_only=True)
        de = self.cnn.reduce_to_d(emb)
        return de


dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def vqc_qnode(x_qubits, theta):
    qml.AngleEmbedding(x_qubits, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(theta, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class VQC_Layer(nn.Module):
    def __init__(self, d_embed=D_EMBED, n_qubits=N_QUBITS, layers=VQC_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.theta = nn.Parameter(torch.zeros(layers, n_qubits, 3))
        nn.init.uniform_(self.theta, -np.pi, np.pi)

        self.pre = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, d_embed, bias=True),
            nn.GELU()
        )
        self.to_qubits = nn.Linear(d_embed, n_qubits)
        self.angle_scale = nn.Parameter(torch.tensor(1.0))
        self.post = nn.Sequential(
            nn.LayerNorm(n_qubits),
            nn.Linear(n_qubits, n_qubits, bias=True),
            nn.GELU()
        )

        nn.init.xavier_uniform_(self.to_qubits.weight)
        if self.to_qubits.bias is not None:
            nn.init.constant_(self.to_qubits.bias, 0)

        for layer in self.pre:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.post:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        orig_shape = x.shape
        is_seq = (x.dim() == 3)
        if is_seq:
            B, T, D = orig_shape
            x = x.view(B * T, D)

        z = self.pre(x)
        xq = self.to_qubits(z)
        scale = 0.5 + F.softplus(self.angle_scale)
        xq = torch.tanh(xq * scale) * (math.pi / 2)

        out = vqc_qnode(xq, self.theta)
        out = torch.stack(out, dim=-1)
        out = out.to(xq.device, dtype=xq.dtype)
        out = self.post(out)

        if is_seq:
            return out.view(B, T, self.n_qubits)
        return out


class Hybrid_Quantum_ASL(nn.Module):
    def __init__(self, features, quantum, head):
        super().__init__()
        self.features = features
        self.quantum = quantum
        self.head = head

    def forward(self, x):
        de = self.features(x)      # [B, D_EMBED]
        q = self.quantum(de)       # [B, N_QUBITS]
        logits = self.head(q)      # [B, NUM_CLASSES]
        return logits


def build_model():
    cnn = ResNetEmbed(emb_dim=E, num_classes=NUM_CLASSES,
                      pretrained=False, freeze_stem=True)
    features = CNNtoDE(cnn)
    quantum = VQC_Layer(d_embed=D_EMBED, n_qubits=N_QUBITS, layers=VQC_LAYERS)
    head = nn.Sequential(
        nn.Linear(N_QUBITS, 32),
        nn.ReLU(),
        nn.Linear(32, NUM_CLASSES),
    )
    model = Hybrid_Quantum_ASL(features, quantum, head)
    return model


def load_image_model(ckpt_path):
    model = build_model().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # state should be the core.state_dict() from training
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    model.eval()
    return model


model = load_image_model(BEST_IMG_PATH)
print("[INFO] Image hybrid model loaded.")


# ====== INFERENCE HELPERS ======

def predict_frame(frame_bgr):
    """frame_bgr: np.array HxWx3 from cv2 (BGR)"""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = transforms.functional.to_pil_image(img_rgb)
    x = val_trsfm_img(pil)              # CxHxW
    x = x.unsqueeze(0).to(DEVICE)       # 1xCxHxW

    with torch.no_grad():
        logits = model(x)               # 1xNUM_CLASSES
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
    return CLASSES[idx], conf


# Build keyboard mapping: 'a' -> CLASSES[0], 'b' -> CLASSES[1], etc.
KEY2CLASS = {}
for i, c in enumerate(CLASSES):
    keycode = ord('a') + i
    if keycode <= ord('z'):
        KEY2CLASS[keycode] = c


def run_webcam(cam_index=0):
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("[ERR] Cannot open webcam")
        return

    print("[INFO] Press 'q' to quit.")
    print("[INFO] Press 'a'.. to START a 3s COUNTDOWN for a labeled segment.")
    print("[INFO] Each segment records for 10s after countdown.")
    print("[INFO] Press 's' to cancel countdown/recording early.")
    print("[INFO] Example: if CLASSES = ['A','B','C'], 'a'->'A', 'b'->'B', 'c'->'C'.")

    history = []
    samples = []   # logs for JSON
    majority_window = 10

    current_true = None
    recording = False
    record_start_time = None

    countdown_active = False
    countdown_start_time = None
    pending_true_label = None

    segment_id = 0  # identify segments

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Failed to grab frame")
            break

        now = time.time()

        pred_class, conf = predict_frame(frame)
        history.append(pred_class)
        if len(history) > majority_window:
            history.pop(0)

        majority_class = Counter(history).most_common(1)[0][0]

        # --- HANDLE COUNTDOWN LOGIC ---
        if countdown_active and countdown_start_time is not None:
            elapsed_cd = now - countdown_start_time
            remaining = COUNTDOWN_SEC - elapsed_cd

            if remaining <= 0.0:
                # Countdown finished -> start recording
                countdown_active = False
                current_true = pending_true_label
                pending_true_label = None
                recording = True
                record_start_time = now
                segment_id += 1
                history.clear()
                print(f"[INFO] START recording segment {segment_id} with TRUE label: {current_true}")
            else:
                # Still counting down, draw overlay
                cd_text = f"Starting {pending_true_label} in {remaining:.1f}s"
                cv2.putText(frame, cd_text, (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

        # --- LOG SAMPLES ONLY WHEN RECORDING ---
        t_rel = 0.0
        if recording and record_start_time is not None:
            t_rel = now - record_start_time

            samples.append({
                "segment": segment_id,
                "t": t_rel,
                "pred": pred_class,
                "conf": conf,
                "maj": majority_class,
                "true": current_true,
            })

            # Auto-stop after fixed duration
            if t_rel >= RECORD_DURATION:
                print(f"[INFO] AUTO-STOP recording segment {segment_id} after {RECORD_DURATION}s")
                recording = False
                current_true = None
                record_start_time = None

        # --- DRAW OVERLAY (prediction + REC state) ---

        text = f"{majority_class} ({conf:.2f})"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        if countdown_active and pending_true_label is not None:
            # redraw countdown text (in case not drawn above due to ordering)
            elapsed_cd = now - countdown_start_time
            remaining = max(0.0, COUNTDOWN_SEC - elapsed_cd)
            cd_text = f"Starting {pending_true_label} in {remaining:.1f}s"
            color = (0, 0, 255)
            cv2.putText(frame, cd_text, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
            rec_text = "REC (countdown)"
        elif recording and current_true is not None:
            rec_text = f"REC {current_true} t={t_rel:.1f}/{RECORD_DURATION:.0f}s"
            color = (0, 0, 255)
            cv2.putText(frame, rec_text, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        else:
            rec_text = "REC OFF"
            color = (128, 128, 128)
            cv2.putText(frame, rec_text, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        cv2.putText(frame, "q: quit | a..: start segment | s: stop",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        cv2.imshow("ASL Quantum Hybrid - Frame Mode", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Start new labeled countdown (only if not already counting down or recording)
        if key in KEY2CLASS:
            if not countdown_active and not recording:
                pending_true_label = KEY2CLASS[key]
                countdown_active = True
                countdown_start_time = now
                history.clear()
                print(f"[INFO] START countdown for label: {pending_true_label}")
            else:
                print("[WARN] Already recording or counting down. Press 's' to stop first.")

        # Stop recording or cancel countdown
        if key == ord("s"):
            if recording:
                print(f"[INFO] MANUAL STOP recording segment {segment_id}")
            if countdown_active:
                print("[INFO] Countdown cancelled.")
            recording = False
            current_true = None
            record_start_time = None
            countdown_active = False
            countdown_start_time = None
            pending_true_label = None

    cap.release()
    cv2.destroyAllWindows()

    # ====== SAVE LOG TO JSON ======
    if samples:
        log_obj = {
            "classes": CLASSES,
            "majority_window": majority_window,
            "countdown_sec": COUNTDOWN_SEC,
            "record_duration_sec": RECORD_DURATION,
            "num_samples": len(samples),
            "samples": samples,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(log_obj, f, indent=2)
        print(f"[INFO] Wrote {len(samples)} samples to {OUT_JSON}")
    else:
        print("[INFO] No samples recorded; nothing saved.")


if __name__ == "__main__":
    run_webcam()
