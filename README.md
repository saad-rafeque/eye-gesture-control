# Eye Gesture Control

Real-time, hands-free cursor control from a webcam. Horizontal gaze moves the cursor, a blink clicks, and a long blink pauses or resumes input. Built with OpenCV, MediaPipe Face Mesh and PyAutoGUI.

This is the vision front end of Sight Switch, my final-year project: smart-home control by eye movement for people with limited mobility. The hardware side of that project (Raspberry Pi, MQTT, ESP32 relays) is not part of this repository.

## Features

- **Gaze left / right** moves the cursor by a set percentage of the screen width
- **Blink** sends a mouse click
- **Long blink** (1 s by default) pauses and resumes all input
- **Debug window** shows the detected gesture and system state live
- Smoothing, stable-frame and cooldown settings to suppress false triggers

## Scripts

| Script | Use it when |
| --- | --- |
| `pupil_tracker.py` | You want the best accuracy. Runs a short calibration at start. |
| `no_calibration.py` | You want to start immediately. Has Linux, macOS and Windows code paths. |
| `no_calibration_windows.py` | Same idea, Windows variant. |

## Quick start

Requires Python 3 and a webcam.

```bash
git clone https://github.com/saad-rafeque/eye-gesture-control.git
cd eye-gesture-control
pip install -r requirements.txt

python pupil_tracker.py     # with calibration
python no_calibration.py    # without calibration
```

Press `q` in the debug window to quit, or `Ctrl+C` in the terminal.

## Configuration

Tunable values sit in the `USER SETTINGS` block at the top of the script.

| Setting | Purpose |
| --- | --- |
| `CAM_INDEX` | Which camera to open |
| `CAM_WIDTH`, `CAM_HEIGHT` | Capture resolution; higher costs more CPU |
| `FPS_LIMIT` | Frame-rate cap |
| `MOVE_PERCENT` | Cursor step as a percentage of screen width |
| `BLINK_THRESHOLD` | Eye-openness ratio below which a blink is counted |
| `LONG_BLINK_DURATION` | Seconds a blink must last to toggle pause |
| `COOLDOWN_SECONDS` | Minimum gap between two commands |

## Troubleshooting

- **Cannot open camera:** check the webcam connection or change `CAM_INDEX`.
- **Low frame rate:** lower `FPS_LIMIT` or the capture resolution.
- **Missed or false blinks:** use even front lighting and adjust `BLINK_THRESHOLD`.

## License

MIT. See [LICENSE](LICENSE).
