# Offline bundle for the Raspberry Pi

Everything the Pi needs. No internet needed for Python 3.11, 3.12 or 3.13:

| Path | What |
|---|---|
| `models/yolo11n_raspberry_executorch_optimized.pte` | Arm YOLO11n **INT8** (4.7 MB), ExecuTorch 1.1.0 |
| `models/yolo11n_raspberry_executorch_original.pte` | Arm YOLO11n **FP32** (10.2 MB), same export |
| `models/yolo11n_fp32_480.pte`, `yolo11n_fp32_320.pte` | our FP32 exports at smaller inputs (Ultralytics, ExecuTorch 1.6 export, loads on 1.1.0) |
| `wheels/` | 29 wheels for Python **3.11, 3.12 and 3.13** (546 MB, not in git: copy to USB) |
| `requirements-pi.txt` | the pinned set `install_pi.sh` installs |
| `SHA256SUMS` | checksums of the wheels |
| `install_pi.sh` | offline install into `.venv-pi` |

Models: Arm AI Portal, `Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5` @ `6e17c67`, AGPL-3.0.
Runtime: executorch 1.1.0, torch 2.10.0 (CPU), numpy 2.2.6, opencv-python-headless 5.0.0.93.

## Known risks
- **Python older than 3.11 or newer than 3.13:** `install_pi.sh` fetches Python 3.13 with `uv` (needs internet).
- **No USB / empty `wheels/`:** `install_pi.sh` installs the same pinned versions from the internet.
- **Pi 4:** no INT8 dot-product instructions; INT8 may be slow or fail, and the stock aarch64 ExecuTorch wheel may assume ARMv8.2. Use the FP32 model and report both.
- The Pi's own numbers are the only ones that count. Laptop timings are for debugging.
