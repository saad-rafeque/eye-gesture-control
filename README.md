## Eye‑Gesture Control

Hands‑free mouse navigation and blink‑to‑click powered by **MediaPipe FaceMesh** and **OpenCV**.  
A hybrid classifier combines a nearest‑centroid approach for left/right gaze with calibrated vertical thresholds for up/down, plus EAR‑based blink detection for clicks.

<div align="center">
  <!-- Replace with your own demo GIF or screenshot -->
  <img src="docs/demo.gif" alt="Eye‑gesture cursor control demo" width="600">
</div>

---

## ✨ Features

| Action | Detection method | Default behaviour |
| ------ | --------------- | ----------------- |
| **Look left / right** | Nearest‑centroid on calibrated iris positions | Move cursor ±100 px (or press ←/→) |
| **Look up / down** | One‑axis threshold (`VERT_FRACTION`) | Move cursor ±100 px |
| **Blink** | Eye‑aspect‑ratio (EAR) below `BLINK_THRESHOLD` | Mouse **click** |
| **Calibration** | 5‑point guided routine (centre, up, left, down, right) | Runs automatically on start‑up |
| **Speech feedback** | `pyttsx3` | Announces direction / “click” |

---

## 🔧 Requirements

* Python 3.9 – 3.12
* Webcam ≥ 640 × 480 @ 30 fps
* Packages:

```txt
mediapipe==0.10.12
opencv‑python==4.10.0.82
numpy==1.26.4
pyautogui==0.9.54
pyttsx3==2.90   # optional – comment out to disable speech
```

Install everything with:

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

---

## 🚀 Quick start

```bash
python eye_gesture_control.py
```

1. **Follow the on‑screen dots** for ~20 seconds while the program learns your neutral and extreme gaze points.  
2. Look in any of the four directions or blink to drive the cursor.  
3. Press **q** (or close the console) to quit.

---

## ⚙️ Configuration

All user‑tunable constants live at the top of `eye_gesture_control.py`.

```python
VERT_FRACTION = 0.35   # Lower → stricter (needs bigger eye roll for up/down)
TOLERANCE = {"left":0.45, "right":0.45}
MOVE_PIXELS = 100      # Cursor step size
USE_ARROW_KEYS = False # True → emit keyboard arrows instead of moving the mouse
```

---

## 🩹 Troubleshooting

| Symptom | Possible fix |
| ------- | ------------ |
| Cursor drifts when looking straight | Reduce `TOLERANCE` or recalibrate |
| Up/down triggers too easily | Lower `VERT_FRACTION` (e.g. 0.25) |
| Clicks fire randomly | Increase `BLINK_THRESHOLD` or `BLINK_COOLDOWN_S` |
| “Camera not open” error | Adjust `CAM_INDEX` (0, 1, 2…) |

---

## 📚 Roadmap

* Continuous cursor velocity instead of fixed steps  
* Dwell‑to‑click mode  
* Packaged installer for Windows/macOS/Linux  

Contributions and feature requests are welcome—open an issue or pull request!

---

## 📝 License

Distributed under the **MIT License**.  
See [`LICENSE`](LICENSE) for the full text.
```
