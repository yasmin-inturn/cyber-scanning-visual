"""
JARVIS // surveillance scanner

Webcam em preto e branco com HUD verde de câmera de segurança,
controlado pelas mãos (MediaPipe Hand Landmarker + OpenCV).
O alvo dentro do scanner ganha uma "leitura comportamental" ao vivo.
É tudo ficção estética: as métricas misturam movimento real com valores inventados.

Gestos:
  2 mãos  -> polegar e indicador de cada mão seguram os 4 cantos do scanner
  1 mão   -> polegar e indicador viram cantos opostos de um retângulo
  polegar + dedo do meio (pinça) -> nova varredura profunda do alvo

Teclas: ESPAÇO varredura | S salva screenshot | C reinicia leitura | Q ou ESC sai
"""
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "hand_landmarker.task"
CAPTURES_DIR = HERE / "captures"
WINDOW = "JARVIS // surveillance scanner"

# Cores em BGR (o OpenCV usa BGR, não RGB)
GREEN = (40, 255, 60)
DARK_GREEN = (12, 50, 14)
AMBER = (0, 190, 255)
WHITE = (235, 235, 235)
BLACK = (0, 0, 0)
RED = (40, 40, 230)
FONT = cv2.FONT_HERSHEY_PLAIN

MARGIN = 28
PANEL_W, PANEL_H = 340, 458
THUMB_W, THUMB_H = 110, 130

# Índices dos 21 pontos da mão no MediaPipe
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 8, 9, 12
FINGER_TIPS = (4, 8, 12, 16, 20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Pinça com histerese: fecha abaixo de PINCH_ON, só solta acima de PINCH_OFF
PINCH_ON, PINCH_OFF = 0.33, 0.50

# Mola dos cantos: STIFFNESS alto = segue mais rápido, DAMPING baixo = balança mais
STIFFNESS, DAMPING = 0.30, 0.62

# ---------------------------------------------------------------- textos da leitura

SCAN_TIME = 2.0  # segundos de "varredura profunda"
METRICS = ("AGITATION", "STRESS INDEX", "FOCUS", "DECEPTION PROB.", "SOCIAL MASKING")

PROFILE_POOLS = {
    "ARCHETYPE": ["THE OBSERVER", "THE STRATEGIST", "THE REBEL", "THE GHOST", "THE ROMANTIC", "THE ARCHITECT"],
    "ATTACHMENT": ["SECURE", "ANXIOUS", "AVOIDANT", "DISORGANIZED"],
    "DEFENSE MECH.": ["SUBLIMATION", "PROJECTION", "DENIAL", "RATIONALIZATION", "HUMOR", "REPRESSION"],
    "CORE DESIRE": ["RECOGNITION", "CONTROL", "FREEDOM", "BELONGING", "ESCAPE"],
    "SHADOW TRAIT": ["ENVY", "NEED FOR APPROVAL", "FEAR OF ABANDONMENT", "PERFECTIONISM", "RESTLESSNESS"],
}

LOG_LINES = [
    lambda: ">> parsing micro-expressions... OK",
    lambda: f">> pupil dilation delta +{random.randint(2, 19)}%",
    lambda: f">> heart rate est. {random.randint(64, 128)} bpm",
    lambda: f">> cross-ref node 0x{random.getrandbits(16):04X}",
    lambda: f">> social graph lookup: {random.randint(3, 480)} links",
    lambda: f">> hand tremor {random.uniform(4, 12):.1f} hz",
    lambda: f">> anomaly score {random.random():.3f}",
    lambda: f">> decrypting behavioral hash {random.getrandbits(32):08x}",
    lambda: ">> voice stress: n/a (no audio feed)",
    lambda: ">> gait signature: partial",
]

THREAT_LEVELS = [
    (0.2, "LOW", GREEN),
    (0.4, "GUARDED", GREEN),
    (0.6, "ELEVATED", AMBER),
    (0.8, "HIGH", RED),
    (1.01, "CRITICAL", RED),
]


def threat_label(value):
    for limit, name, color in THREAT_LEVELS:
        if value < limit:
            return name, color
    return THREAT_LEVELS[-1][1:]


def roll_profile():
    lines = [f"{key}: {random.choice(options)}" for key, options in PROFILE_POOLS.items()]
    eros = random.randint(25, 85)
    lines.insert(2, f"DRIVE: EROS {eros}% / THANATOS {100 - eros}%")
    return lines


def scramble(n):
    return "".join(random.choice("#$%&@01ABCDEF/\\") for _ in range(n))


# ---------------------------------------------------------------- tracking

class HandTracker:
    def __init__(self, model_path, max_hands=2):
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ts = 0

    def detect(self, frame_bgr):
        """Retorna uma lista de mãos; cada mão é uma lista de 21 pontos (x, y) em pixels."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # O modo VIDEO exige timestamps sempre crescentes
        ts = max(int(time.monotonic() * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        h, w = frame_bgr.shape[:2]
        return [[(int(p.x * w), int(p.y * h)) for p in hand] for hand in result.hand_landmarks]

    def close(self):
        self.landmarker.close()


class Spring:
    """Canto que persegue o dedo como uma mola: suaviza o tremor e dá um leve 'balanço'."""

    def __init__(self):
        self.pos = None
        self.vel = np.zeros(2, np.float32)

    def update(self, target):
        target = np.asarray(target, np.float32)
        if self.pos is None:
            self.pos = target.copy()
        else:
            self.vel = (self.vel + (target - self.pos) * STIFFNESS) * DAMPING
            self.pos = self.pos + self.vel
        return self.pos

    def reset(self):
        self.pos = None
        self.vel = np.zeros(2, np.float32)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def pinch_ratio(pts, finger_tip):
    # Normaliza pelo tamanho da mão, assim funciona perto ou longe da câmera
    hand_size = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0
    return dist(pts[THUMB_TIP], pts[finger_tip]) / hand_size


def corner_targets(hands):
    """4 alvos com identidade fixa (cada um é sempre o mesmo dedo)."""
    if len(hands) >= 2:
        left, right = sorted(hands[:2], key=lambda p: p[WRIST][0])
        return [left[INDEX_TIP], right[INDEX_TIP], right[THUMB_TIP], left[THUMB_TIP]]
    (ax, ay), (bx, by) = hands[0][THUMB_TIP], hands[0][INDEX_TIP]
    x1, x2 = sorted((ax, bx))
    y1, y2 = sorted((ay, by))
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def quad_order(pts):
    """Ordem dos cantos em volta do centro, começando pelo de cima/esquerda (TL, TR, BR, BL)."""
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    start = int(np.argmin(pts[order].sum(axis=1)))
    return np.roll(order, -start)


def fluid_outline(quad, vels, t, samples=20):
    """Bordas curvas: cada lado entorta pra trás quando é arrastado e ondula de leve."""
    pts = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        length = float(np.linalg.norm(edge)) + 1e-6
        normal = np.array([edge[1], -edge[0]], np.float32) / length
        drag = -float(np.dot((vels[i] + vels[(i + 1) % 4]) / 2, normal)) * 2.0
        wobble = np.sin(t * 2.4 + i * 1.7) * min(6.0, length * 0.03)
        bulge = np.clip(drag + wobble, -0.25 * length, 0.25 * length)
        ctrl = (a + b) / 2 + normal * bulge
        s = np.linspace(0, 1, samples, endpoint=False)[:, None]
        pts.append((1 - s) ** 2 * a + 2 * (1 - s) * s * ctrl + s ** 2 * b)
    return np.vstack(pts).astype(np.int32)


# ---------------------------------------------------------------- efeitos

def build_film_mask(w, h):
    """Vinheta + scanlines pré-calculadas (multiplicam o frame em P&B)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vignette = np.clip(1.15 - 0.45 * d2, 0.35, 1.0)
    scanlines = np.where(np.arange(h) % 3 == 0, 0.82, 1.0).astype(np.float32)[:, None]
    return vignette * scanlines


def surveillance_filter(frame, mask, noise):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray - 128) * 1.35 + 128  # mais contraste
    gray = gray * mask + noise         # vinheta, scanlines e granulado
    return np.clip(gray, 0, 255).astype(np.uint8)


def make_lut(dark, light):
    t = np.linspace(0, 1, 256)[:, None]
    return (np.array(dark) * (1 - t) + np.array(light) * t).astype(np.uint8)


GREEN_LUT = make_lut(DARK_GREEN, GREEN)


def green_tint(gray):
    return GREEN_LUT[gray]


def halftone(gray, cell=6):
    """Retícula verde estilo pôster: pontos escuros maiores onde a imagem é mais escura."""
    h, w = gray.shape
    out = np.full((h, w, 3), GREEN, np.uint8)
    small = cv2.resize(gray, (w // cell, h // cell), interpolation=cv2.INTER_AREA)
    r_max = cell * 0.72
    for j in range(small.shape[0]):
        for i in range(small.shape[1]):
            r = (1 - small[j, i] / 255.0) * r_max
            if r > 0.6:
                center = (i * cell + cell // 2, j * cell + cell // 2)
                cv2.circle(out, center, int(round(r)), DARK_GREEN, -1, cv2.LINE_AA)
    return out


def warp_quad(gray, quad, w, h):
    """'Desentorta' o formato pra um retângulo, como uma foto tirada de frente."""
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(gray, matrix, (w, h))


# ---------------------------------------------------------------- desenho

def text(img, s, org, color=GREEN, scale=1.0):
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def tag(img, s, x, y, fg=BLACK, bg=GREEN, scale=1.0, pad=4, width=None):
    """Etiqueta com fundo sólido (tipo 'ID: P-2204-02'). Retorna o y da próxima linha."""
    (tw, th), base = cv2.getTextSize(s, FONT, scale, 1)
    w = width or tw + 2 * pad
    cv2.rectangle(img, (x, y), (x + w, y + th + base + 2 * pad), bg, -1)
    text(img, s, (x + pad, y + th + pad), fg, scale)
    return y + th + base + 2 * pad + 4


def bar(img, x, y, w, value, color=GREEN, h=8):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    fill = int((w - 2) * float(np.clip(value, 0, 1)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)


def dashed_rect(img, p1, p2, color, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 1)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 1)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 1)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 1)


def draw_hand(img, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], GREEN, 1, cv2.LINE_AA)
    for i in FINGER_TIPS:
        cv2.circle(img, pts[i], 3, GREEN, -1, cv2.LINE_AA)
    x, y = pts[WRIST]
    dashed_rect(img, (x - 16, y - 16), (x + 16, y + 16), GREEN)


def draw_scanner(img, gray, quad, vels, t, locked, flash, label):
    h, w = gray.shape
    outline = fluid_outline(quad, vels, t)
    x, y, bw, bh = cv2.boundingRect(outline)
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    # Máscara suave (anti-aliasing) com o formato fluido, só na região do scanner
    local = outline - np.array([x0, y0], np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [local], 255, cv2.LINE_AA)
    alpha = mask[..., None].astype(np.float32) / 255.0

    roi = img[y0:y1, x0:x1]
    tinted = green_tint(gray[y0:y1, x0:x1]).astype(np.float32)
    if flash:
        tinted = tinted * 0.3 + 255 * 0.7
    roi[:] = (roi * (1 - alpha) + tinted * alpha).astype(np.uint8)

    # Brilho em volta da borda
    glow = roi.copy()
    cv2.polylines(glow, [local], True, GREEN, 8, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.25, roi, 0.75, 0, dst=roi)

    # Linha de varredura subindo e descendo, recortada pelo formato
    y_line = y + int(abs(((t * 0.9) % 2) - 1) * max(bh - 1, 0))
    if y0 <= y_line < y1:
        inside = mask[y_line - y0] > 128
        img[y_line, x0:x1][inside] = WHITE

    cv2.polylines(img, [outline], True, GREEN, 2 if locked else 1, cv2.LINE_AA)

    # Cantos "agarrados" pelos dedos
    for i, (cx, cy) in enumerate(quad.astype(int)):
        r = int(10 + 3 * np.sin(t * 5 + i * 1.3))
        cv2.circle(img, (cx, cy), r, GREEN, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, WHITE, -1, cv2.LINE_AA)
        text(img, f"{max(cx, 0):04d}.{max(cy, 0):04d}", (cx + 14, cy - 10), GREEN, 0.7)

    # Mira no centro
    mx, my = quad.mean(axis=0).astype(int)
    cv2.line(img, (mx - 8, my), (mx + 8, my), GREEN, 1)
    cv2.line(img, (mx, my - 8), (mx, my + 8), GREEN, 1)

    text(img, label, (x, min(y + bh + 20, h - 4)), GREEN, 0.9)


# ---------------------------------------------------------------- leitura comportamental

class BehaviorReader:
    """Painel de 'leitura do alvo': métricas ao vivo, perfil psicológico e nível de perigo.

    Agitação e estresse vêm de dados reais (movimento dentro do scanner e velocidade
    das mãos); o resto é teatro, com variação aleatória pra parecer vivo.
    """

    def __init__(self):
        self.values = {k: 0.0 for k in METRICS}
        self.walk = {"DECEPTION PROB.": 0.3, "SOCIAL MASKING": 0.5}
        self.threat = 0.0
        self.bias = 0.0
        self.match = 0
        self.profile = []
        self.subject_id = "--"
        self.scan_start = None
        self.reveal_start = 0.0
        self.log = deque(maxlen=7)
        self.next_log = 0.0
        self.last_seen = -10.0
        self.prev_small = None
        self.thumb = None
        self.frame_n = 0

    def active(self, now):
        return now - self.last_seen < 1.5

    def scanning(self, now):
        return self.scan_start is not None and now - self.scan_start < SCAN_TIME

    def start_scan(self, now):
        self.scan_start = now
        self.subject_id = f"P-{random.randint(1000, 9999)}-{random.randint(1, 99):02d}"
        self.log.append(">> LOCK ACQUIRED // deep scan initiated")

    def update(self, raw_gray, gray, quad, vels, now):
        self.frame_n += 1
        small = cv2.GaussianBlur(cv2.resize(raw_gray, (160, 90)), (5, 5), 0)
        prev, self.prev_small = self.prev_small, small
        if quad is None:
            for k in self.values:
                self.values[k] *= 0.95
            return

        if not self.active(now):
            self.start_scan(now)
        self.last_seen = now

        # Movimento real dentro do scanner (diferença entre frames, em baixa resolução)
        agitation = 0.0
        if prev is not None:
            scale = np.array([160 / raw_gray.shape[1], 90 / raw_gray.shape[0]], np.float32)
            x, y, w, h = cv2.boundingRect((quad * scale).astype(np.int32))
            x, y = max(0, x), max(0, y)
            region_now, region_prev = small[y:y + h, x:x + w], prev[y:y + h, x:x + w]
            if region_now.size:
                agitation = min(1.0, float(cv2.absdiff(region_now, region_prev).mean()) / 12)
        speed = min(1.0, float(np.linalg.norm(vels, axis=1).mean()) / 20)

        for k in self.walk:
            self.walk[k] = float(np.clip(self.walk[k] + random.uniform(-0.03, 0.03), 0.05, 0.95))
        targets = {
            "AGITATION": agitation,
            "STRESS INDEX": 0.7 * speed + 0.3 * agitation,
            "FOCUS": 1 - 0.8 * max(agitation, speed),
            "DECEPTION PROB.": self.walk["DECEPTION PROB."],
            "SOCIAL MASKING": self.walk["SOCIAL MASKING"],
        }
        for k, v in targets.items():
            self.values[k] += 0.12 * (v + random.uniform(-0.03, 0.03) - self.values[k])

        v = self.values
        threat = 0.4 * v["AGITATION"] + 0.3 * v["STRESS INDEX"] + 0.15 * v["DECEPTION PROB."] + self.bias
        self.threat += 0.05 * (float(np.clip(threat, 0, 1)) - self.threat)

        if self.scan_start is not None and not self.scanning(now):
            self.scan_start = None
            self.profile = roll_profile()
            self.bias = random.uniform(0, 0.35)
            self.match = random.randint(61, 99)
            self.reveal_start = now
            self.log.append(">> profile compiled // threat matrix updated")

        if now >= self.next_log:
            self.log.append(random.choice(LOG_LINES)())
            self.next_log = now + (0.12 if self.scanning(now) else random.uniform(0.4, 1.0))

        self.thumb = halftone(cv2.equalizeHist(warp_quad(gray, quad, THUMB_W, THUMB_H)), cell=5)

    def scanner_label(self):
        name, _ = threat_label(self.threat)
        return f"subject {self.subject_id} // threat: {name}"

    def draw(self, img, now, quad):
        if not self.active(now):
            return
        h, w = img.shape[:2]
        x0, y = MARGIN + 16, MARGIN + 16

        # Fundo escurecido pra dar leitura
        roi = img[y - 6:y + PANEL_H, x0 - 6:x0 + PANEL_W + 6]
        roi[:] = (roi * 0.3).astype(np.uint8)

        y = tag(img, "BEHAVIORAL READ // SUBJECT ANALYSIS", x0, y, width=PANEL_W)
        if int(now * 3) % 2 == 0:
            cv2.circle(img, (x0 + PANEL_W - 12, y - 15), 4, RED, -1, cv2.LINE_AA)

        # Miniatura do alvo + dados de identificação
        if self.thumb is not None:
            img[y:y + THUMB_H, x0:x0 + THUMB_W] = self.thumb
        cv2.rectangle(img, (x0 - 1, y - 1), (x0 + THUMB_W, y + THUMB_H), GREEN, 1)
        if quad is not None:
            cv2.line(img, (x0 + THUMB_W, y + 20), tuple(int(v) for v in quad[0]), GREEN, 1, cv2.LINE_AA)

        tx = x0 + THUMB_W + 12
        scanning = self.scanning(now)
        if scanning:
            pct = (now - self.scan_start) / SCAN_TIME
            status, status_color = f"DEEP SCAN {int(pct * 100):3d}%", WHITE
            match = f"{random.randint(10, 99)}%"
        else:
            status, status_color = "PROFILE LOCKED", GREEN
            match = f"{self.match}%"
        rows = [
            (f"ID: {self.subject_id}", WHITE),
            (f"STATUS: {status}", status_color),
            (f"BIOMETRIC MATCH: {match}", GREEN),
            (f"FRAME: {self.frame_n:06d}", GREEN),
            ("NODE: CAM_01 // SECTOR 7", GREEN),
        ]
        for i, (s, color) in enumerate(rows):
            text(img, s, (tx, y + 14 + i * 18), color, 0.85)
        if scanning:
            bar(img, tx, y + 104, PANEL_W - THUMB_W - 12, pct, WHITE)
        y += THUMB_H + 22

        # Métricas comportamentais
        text(img, "// BEHAVIORAL METRICS", (x0, y), GREEN, 0.9)
        y += 10
        for k in METRICS:
            value = self.values[k]
            color = RED if value > 0.75 else GREEN
            text(img, k, (x0, y + 9), WHITE, 0.8)
            bar(img, x0 + 130, y + 1, 150, value, color)
            text(img, f"{int(value * 100):3d}%", (x0 + 290, y + 9), color, 0.8)
            y += 18
        y += 16

        # Perfil psicológico (embaralhado durante a varredura, depois "digitado")
        text(img, "// PSYCH PROFILE", (x0, y), GREEN, 0.9)
        y += 18
        if scanning or not self.profile:
            for key in PROFILE_POOLS:
                text(img, f"{key}: {scramble(12)}", (x0, y), DARK_GREEN if random.random() < 0.3 else GREEN, 0.8)
                y += 16
            y += 16
        else:
            budget = int((now - self.reveal_start) * 70)
            for line in self.profile:
                shown = line[:max(0, budget)]
                cursor = "_" if 0 < budget < len(line) and int(now * 6) % 2 == 0 else ""
                text(img, shown + cursor, (x0, y), WHITE, 0.8)
                budget -= len(line)
                y += 16
        y += 10

        # Nível de perigo
        name, color = threat_label(self.threat)
        text(img, "THREAT LEVEL", (x0, y), WHITE, 0.9)
        text(img, f"{self.threat * 100:5.1f} / 100", (x0 + 240, y), color, 0.9)
        y += 8
        filled = int(round(self.threat * 10))
        for i in range(10):
            sx = x0 + i * 25
            cv2.rectangle(img, (sx, y), (sx + 20, y + 14), color, -1 if i < filled else 1)
        blink_off = name == "CRITICAL" and int(now * 4) % 2 == 0
        if not blink_off:
            tag(img, name, x0 + 255, y - 2, bg=color, scale=0.9, width=PANEL_W - 255)

        # Feed de log no canto inferior direito
        lx = w - MARGIN - 300
        ly = h - MARGIN - 20 - 16 * (len(self.log) - 1)
        log_bg = img[ly - 16:h - MARGIN - 8, lx - 8:w - MARGIN - 8]
        log_bg[:] = (log_bg * 0.3).astype(np.uint8)
        for i, line in enumerate(self.log):
            fade = (i + 1) / len(self.log)
            color = tuple(int(c * (0.35 + 0.65 * fade)) for c in GREEN)
            text(img, line, (lx, ly + i * 16), color, 0.8)


def draw_hud(img, t, fps, n_hands):
    h, w = img.shape[:2]
    m = MARGIN
    cv2.rectangle(img, (m, m), (w - m, h - m), WHITE, 1)

    text(img, "CAM_01 // surveillance node active", (m, m - 9))
    stamp = datetime.now().strftime("%Y.%m.%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(stamp, FONT, 1.0, 1)
    text(img, stamp, (w - m - tw, m - 9))

    # Régua inferior
    yb = h - m
    for x in range(m, w - m, 80):
        cv2.line(img, (x, yb), (x, yb - 8), WHITE, 1)
    cv2.circle(img, (w // 2, yb), 6, WHITE, 1, cv2.LINE_AA)

    text(img, "signal strength: 92%", (m, h - 9))
    mid = f"hands: {n_hands}  //  noise level: minimal"
    (tw, _), _ = cv2.getTextSize(mid, FONT, 1.0, 1)
    text(img, mid, ((w - tw) // 2, h - 9))
    right = f"output filter: active  {fps:4.0f} fps"
    (tw, _), _ = cv2.getTextSize(right, FONT, 1.0, 1)
    text(img, right, (w - m - tw, h - 9))

    if int(t * 2) % 2 == 0:  # REC piscando
        cv2.circle(img, (w - m - 52, m + 22), 6, RED, -1, cv2.LINE_AA)
    text(img, "REC", (w - m - 40, m + 27), WHITE)


# ---------------------------------------------------------------- loop

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Modelo não encontrado em {MODEL_PATH}.\n"
            "Baixe com:\n  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(
            "Não consegui abrir a câmera. No macOS, libere o acesso em "
            "Ajustes do Sistema > Privacidade e Segurança > Câmera para o app do terminal."
        )

    h, w = frame.shape[:2]
    mask = build_film_mask(w, h)
    noises = [np.random.normal(0, 9, (h, w)).astype(np.float32) for _ in range(6)]

    tracker = HandTracker(MODEL_PATH)
    springs = [Spring() for _ in range(4)]
    reader = BehaviorReader()
    pinching = False
    scan_requested = False
    flash_until = 0.0
    fps, prev = 0.0, time.time()
    frame_i = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espelho: mexer a mão pra direita move pra direita
            now = time.time()

            hands = tracker.detect(frame)
            raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = surveillance_filter(frame, mask, noises[frame_i % len(noises)])
            frame_i += 1
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            quad = quad_vel = None
            if hands:
                pos = np.array([s.update(p) for s, p in zip(springs, corner_targets(hands))])
                vel = np.array([s.vel for s in springs])
                order = quad_order(pos)
                quad, quad_vel = pos[order], vel[order]

                threshold = PINCH_OFF if pinching else PINCH_ON
                now_pinching = min(pinch_ratio(p, MIDDLE_TIP) for p in hands) < threshold
                if ((now_pinching and not pinching) or scan_requested) and not reader.scanning(now):
                    reader.start_scan(now)
                    flash_until = now + 0.15
                pinching = now_pinching
            else:
                for s in springs:
                    s.reset()
                pinching = False

            reader.update(raw_gray, gray, quad, quad_vel, now)

            if quad is not None:
                draw_scanner(out, gray, quad, quad_vel, now, pinching, now < flash_until, reader.scanner_label())
                for pts in hands:
                    draw_hand(out, pts)
            elif not reader.active(now) and int(now * 1.5) % 2 == 0:
                msg = "NO SUBJECT // RAISE HANDS TO SCAN"
                (tw, _), _ = cv2.getTextSize(msg, FONT, 1.4, 1)
                text(out, msg, ((w - tw) // 2, h // 2), GREEN, 1.4)

            reader.draw(out, now, quad)

            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1 / dt)
            draw_hud(out, now, fps, len(hands))

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            scan_requested = key == ord(" ")
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                CAPTURES_DIR.mkdir(exist_ok=True)
                path = CAPTURES_DIR / f"scan_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), out)
                print(f"salvo: {path}")
            if key == ord("c"):
                reader = BehaviorReader()
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
