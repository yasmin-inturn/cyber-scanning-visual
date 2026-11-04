# JARVIS // surveillance scanner (versão visual)

Webcam em preto e branco com HUD verde de câmera de segurança, controlada pelas mãos com
MediaPipe e OpenCV. O scanner tem bordas fluidas e abre um painel de "leitura comportamental" do alvo.

**É tudo ficção estética.** Agitação e estresse vêm do movimento real na imagem e das mãos;
perfil psicológico, batimentos, "biometric match" e os demais dados são inventados. O programa
não reconhece rostos nem identifica pessoas.

A versão com leitura facial real e gravação de evidências está em um projeto separado.

## Como rodar

Testado com Python 3.14 no macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
python jarvis_scanner.py
```

No macOS, libere a câmera para o PyCharm ou o Terminal em
Ajustes do Sistema > Privacidade e Segurança > Câmera.

## Controles

| Gesto | Ação |
| --- | --- |
| 2 mãos | Polegar e indicador de cada mão seguram os 4 cantos do scanner |
| 1 mão | Polegar e indicador viram cantos opostos de um retângulo |
| Pinça polegar + médio | Nova varredura do alvo |

| Tecla | Ação |
| --- | --- |
| `Espaço` | Nova varredura (com as mãos na imagem) |
| `S` | Salva um screenshot em `captures/` |
| `C` | Reinicia a leitura |
| `Q` ou `Esc` | Sai |
