# Offline bundle for the Raspberry Pi

Everything the Pi needs, no internet:

| Path | What |
|---|---|
| `models/yolo11n_raspberry_executorch_optimized.pte` | Arm YOLO11n **INT8** (4.7 MB), ExecuTorch 1.1.0 |
| `models/yolo11n_raspberry_executorch_original.pte` | Arm YOLO11n **FP32** (10.2 MB), same export |
| `wheels/` | 18 aarch64 wheels, Python **3.13** (not in git: copy to USB) |
| `SHA256SUMS` | checksums of the wheels |
| `install_pi.sh` | offline install into `.venv-pi` |

Models: Arm AI Portal, `Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5` @ `6e17c67`, AGPL-3.0.
Runtime: executorch 1.1.0, torch 2.10.0 (CPU), numpy 2.2.6, opencv-python-headless 5.0.0.93.

## Known risks
- **Python is not 3.13** (Pi OS Bookworm ships 3.11): these wheels will not install. Ask for a 3.13 env (`uv python install 3.13` needs internet) or rebuild the wheels for the Pi's version.
- **Pi 4:** no INT8 dot-product instructions; INT8 may be slow or fail, and the stock aarch64 ExecuTorch wheel may assume ARMv8.2. Use the FP32 model and report both.
- The Pi's own numbers are the only ones that count. Laptop timings are for debugging.
