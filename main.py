import os, math, cv2, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torch.serialization import add_safe_globals
from PIL import Image
import pennylane as qml

CLASSES       = ["A","B","C","D","E"]   # must match training order
IMG_SIZE      = (112, 112)
E             = 128     # ResNet head emb dim
D_EMBED       = 8       # reduced embedding fed to VQC
N_QUBITS      = 4
VQC_LAYERS    = 2
HIDDEN_LAYER  = 64
T_FRAMES      = 6       # rolling window size
STRIDE        = 3       # VQC stride used in training
DEVICE_TYPE   = "lightning.qubit"  # or "lightning.gpu" if installed
INTERFACE     = "torch"
DIFF_METHOD   = "adjoint"          # not used for grads in inference, but OK
IMG_CKPT      = "best_img.pth"
SEQ_BUNDLE    = "final_bundle_seq.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Models (inference-only)
# -----------------------------
class ResNetEmbed(nn.Module):
    def __init__(self, emb_dim=E, num_classes=len(CLASSES), pretrained=False, freeze_stem=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_stem:
            for name, p in self.backbone.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1")):
                    p.requires_grad = False

        self.proj = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU(inplace=True))
        self.head_cls = nn.Linear(emb_dim, num_classes)  # unused at inference
        self.reduce_to_d = nn.Linear(emb_dim, D_EMBED)   # must match D_EMBED

    def forward(self, x, return_embed_only=False):
        feat512 = self.backbone(x)
        emb = self.proj(feat512)
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
        de  = self.cnn.reduce_to_d(emb)
        return de

class VQC_Layer(nn.Module):
    """
    Variational quantum block:
      pre -> to_qubits (angle prep) -> QNode -> post
    QNode is created inside __init__, no globals.
    """
    def __init__(self, d_embed=D_EMBED, n_qubits=N_QUBITS, layers=VQC_LAYERS,
                 device_type=DEVICE_TYPE, interface=INTERFACE, diff_method=DIFF_METHOD):
        super().__init__()
        self.n_qubits = n_qubits

        # classical parts
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
                nn.init.xavier_uniform_(layer.weight); nn.init.constant_(layer.bias, 0)
        for layer in self.post:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight); nn.init.constant_(layer.bias, 0)

        # quantum params
        import math as _math
        self.theta = nn.Parameter(torch.zeros(layers, n_qubits, 3))
        with torch.no_grad():
            self.theta.uniform_(-_math.pi, _math.pi)

        # make a device and bind qnode
        self._dev = qml.device(device_type, wires=n_qubits)

        @qml.qnode(self._dev, interface=interface, diff_method=diff_method)
        def _qnode(x_qubits, theta):
            qml.AngleEmbedding(x_qubits, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(theta, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._qnode = _qnode  # bound method-like closure

    def forward(self, x):
        # x: (B,D) or (B,T,D)
        is_seq = x.dim() == 3
        if is_seq:
            B, T, D = x.shape
            x = x.view(B*T, D)

        z  = self.pre(x)
        xq = self.to_qubits(z)
        scale = 0.5 + F.softplus(self.angle_scale)
        xq = torch.tanh(xq * scale) * (math.pi/2)

        # Pennylane QNodes return torch tensors with shape (n_qubits,)
        out = self._qnode(xq, self.theta)          # -> (n_qubits,) * (B or B*T) internally
        out = torch.stack(out, dim=-1)             # (B or B*T, n_qubits)
        out = out.to(xq.device, dtype=xq.dtype)
        out = self.post(out)

        if is_seq:
            return out.view(B, T, self.n_qubits)
        return out

# QLSTM with 4 separate QNodes (f, i, u, o)
def _make_qnode(n_qubits, device_type=DEVICE_TYPE, interface=INTERFACE, diff_method=DIFF_METHOD):
    dev = qml.device(device_type, wires=n_qubits)
    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def _q(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
    return _q

class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=N_QUBITS, n_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits

        self.fc = nn.Sequential(
            nn.Linear(input_size, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, n_qubits, bias=True)
        )
        self.input_gain = nn.Parameter(torch.ones(n_qubits) * (math.pi/2))

        # quantum weights
        self.f_weight = nn.Parameter(torch.empty(n_layers, n_qubits, 3))
        self.i_weight = nn.Parameter(torch.empty(n_layers, n_qubits, 3))
        self.u_weight = nn.Parameter(torch.empty(n_layers, n_qubits, 3))
        self.o_weight = nn.Parameter(torch.empty(n_layers, n_qubits, 3))
        with torch.no_grad():
            for p in (self.f_weight, self.i_weight, self.u_weight, self.o_weight):
                p.uniform_(-math.pi, math.pi)

        self.q2hidden = nn.Sequential(
            nn.Linear(n_qubits, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.forget_bias = nn.Parameter(torch.ones(hidden_size))
        self.output = nn.Sequential(nn.Linear(hidden_size, 1, bias=True))

        # four QNodes
        self.qforget = _make_qnode(n_qubits)
        self.qinput  = _make_qnode(n_qubits)
        self.qupdate = _make_qnode(n_qubits)
        self.qoutput = _make_qnode(n_qubits)

    def forward(self, x):  # x: (B,T,n_qubits)
        B, T, _ = x.shape
        device = x.device
        h_t = torch.zeros(B, self.hidden_size, device=device)
        c_t = torch.zeros(B, self.hidden_size, device=device)

        for t in range(T):
            x_t = self.fc(x[:, t, :]) * self.input_gain

            f = torch.stack(self.qforget(x_t, self.f_weight), dim=-1).to(device, x_t.dtype)
            i = torch.stack(self.qinput( x_t, self.i_weight), dim=-1).to(device, x_t.dtype)
            u = torch.stack(self.qupdate(x_t, self.u_weight), dim=-1).to(device, x_t.dtype)
            o = torch.stack(self.qoutput(x_t, self.o_weight), dim=-1).to(device, x_t.dtype)

            f = torch.sigmoid(self.q2hidden(f) + self.forget_bias)
            i = torch.sigmoid(self.q2hidden(i))
            o = torch.sigmoid(self.q2hidden(o))
            u = torch.tanh(   self.q2hidden(u))

            c_t = f * c_t + i * u
            h_t = o * torch.tanh(c_t)

        y = self.output(h_t)
        return y, h_t

class TimeWiseVQC(nn.Module):
    def __init__(self, vqc_layer: VQC_Layer, stride=STRIDE):
        super().__init__()
        self.vqc = vqc_layer
        self.stride = stride
    def forward(self, de_seq):  # (B,T,D_EMBED)
        B, T, D = de_seq.shape
        idx = torch.arange(0, T, self.stride, device=de_seq.device)
        x_sub = de_seq[:, idx, :]
        q_coarse = self.vqc(x_sub)                            # (B, T/stride, n_qubits)
        q_full   = torch.repeat_interleave(q_coarse, repeats=self.stride, dim=1)
        q_full   = q_full[:, :T, :]
        return q_full

class Hybrid_Quantum_ALS_Sequence(nn.Module):
    def __init__(self, vqc_seq: TimeWiseVQC, qlstm_cell: QLSTMCell,
                 num_classes=len(CLASSES), hidden=HIDDEN_LAYER):
        super().__init__()
        self.vqc_seq = vqc_seq
        self.qlstm   = qlstm_cell
        self.head    = nn.Linear(hidden, num_classes)
    def forward(self, de_seq):  # (B,T,D_EMBED)
        q_seq = self.vqc_seq(de_seq)
        _, h_t = self.qlstm(q_seq)
        logits = self.head(h_t)
        return logits

# -----------------------------
# Build models and load weights
# -----------------------------
def build_feature_model():
    cnn = ResNetEmbed(emb_dim=E, num_classes=len(CLASSES), pretrained=False, freeze_stem=True)
    cnn.reduce_to_d = nn.Linear(E, D_EMBED)
    feat = CNNtoDE(cnn)
    return feat

def build_sequence_model():
    vqc = VQC_Layer(d_embed=D_EMBED, n_qubits=N_QUBITS, layers=VQC_LAYERS)
    vqc_seq = TimeWiseVQC(vqc, stride=STRIDE)
    qlstm = QLSTMCell(input_size=N_QUBITS, hidden_size=HIDDEN_LAYER, n_qubits=N_QUBITS, n_layers=2)
    seq = Hybrid_Quantum_ALS_Sequence(vqc_seq, qlstm, num_classes=len(CLASSES), hidden=HIDDEN_LAYER)
    return seq

@torch.no_grad()
def _load_feature_weights(feature, img_ckpt, device):
    """Load CNN->DE weights, remapping keys from 'features.cnn.*' if needed."""
    if not os.path.exists(img_ckpt):
        print(f"[WARN] {img_ckpt} not found; feature model random init.")
        return
    try:
        raw = torch.load(img_ckpt, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(img_ckpt, map_location=device)
    except Exception as e:
        print(f"[WARN] Cannot load {img_ckpt}: {e}")
        return

    if isinstance(raw, dict):
        # If keys look like 'features.cnn.backbone.*', strip that prefix
        if any(k.startswith("features.cnn.") for k in raw.keys()):
            feat_state = {k.replace("features.cnn.", "cnn."): v
                          for k, v in raw.items() if k.startswith("features.cnn.")}
            missing, unexpected = feature.load_state_dict(feat_state, strict=False)
        # Otherwise try loading directly
        else:
            missing, unexpected = feature.load_state_dict(raw, strict=False)
        if missing:
            print("[INFO] Missing feature weights:", missing[:8], "…")
        if unexpected:
            print("[INFO] Unexpected feature weights:", unexpected[:8], "…")
    else:
        print("[WARN] Unsupported checkpoint format for", img_ckpt)

@torch.no_grad()
def _load_seq_weights(seq_model, seq_bundle, device):
    """Load sequence (VQC+QLSTM) weights safely under PyTorch ≥ 2.6."""
    if not os.path.exists(seq_bundle):
        print(f"[WARN] {seq_bundle} not found; sequence model random init.")
        return

    # allow-list classes for unpickling (trusted checkpoint)
    add_safe_globals([Hybrid_Quantum_ALS_Sequence, TimeWiseVQC, VQC_Layer, QLSTMCell])

    try:
        pkg = torch.load(seq_bundle, map_location=device, weights_only=False)
    except TypeError:
        pkg = torch.load(seq_bundle, map_location=device)

    state = None
    if isinstance(pkg, dict):
        ckpt = pkg.get("checkpoint")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        elif all(isinstance(k, str) and torch.is_tensor(v) for k, v in pkg.items()):
            state = pkg

    if state is None:
        raise RuntimeError("No model_state found in final bundle; re-export with state_dict next time.")

    missing, unexpected = seq_model.load_state_dict(state, strict=False)
    if missing:
        print("[INFO] Missing seq weights:", missing[:8], "…")
    if unexpected:
        print("[INFO] Unexpected seq weights:", unexpected[:8], "…")

@torch.no_grad()
def load_models(img_ckpt=IMG_CKPT, seq_bundle=SEQ_BUNDLE):
    """Full loader for webcam inference."""
    feature = build_feature_model().to(device).eval()
    seq     = build_sequence_model().to(device).eval()

    _load_feature_weights(feature, img_ckpt, device)
    _load_seq_weights(seq, seq_bundle, device)

    return feature, seq
# -----------------------------
# Webcam loop
# -----------------------------
def _make_img_transform():
    H,W = IMG_SIZE
    return T.Compose([
        T.Resize((H,W)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

@torch.no_grad()
def run_webcam():
    feature, seq = load_models(IMG_CKPT, SEQ_BUNDLE)
    feature.eval(); seq.eval()

    tfm = _make_img_transform()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam"); return

    buffer = []  # rolling DE frames
    WIN = T_FRAMES

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = tfm(pil).unsqueeze(0).to(device)  # (1,3,H,W)

            # DE frame
            de = feature(x)           # (1, D_EMBED)
            buffer.append(de.squeeze(0))  # (D_EMBED,)
            if len(buffer) > WIN:
                buffer = buffer[-WIN:]

            # Only predict when we have WIN frames
            if len(buffer) == WIN:
                de_seq = torch.stack(buffer, dim=0).unsqueeze(0).to(device)  # (1,T,D_EMBED)
                logits = seq(de_seq)
                pred = int(logits.argmax(dim=1).item())
                label = CLASSES[pred]
                cv2.putText(frame, f"{label}", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("ASL-TQAI (Webcam)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_webcam()
