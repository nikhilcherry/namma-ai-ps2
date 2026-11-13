---
license: agpl-3.0
pipeline_tag: object-detection
library_name: executorch
base_model: Ultralytics/YOLO11
base_model_relation: quantized
tags:
  - arm
  - arm-optimized
  - edge-linux
  - executorch
  - int8
  - coco
  - object-detection
  - xnnpack
---

# YOLO11n optimized for Arm-based Edge Linux

YOLO11n object detection optimized using static post-training quantization (PTQ), with INT8 quantization across eligible backbone and neck operations while the stem convolution (model.0.conv) and complete detection head (model.23) remain in FP32 to preserve accuracy within acceptable ranges. The model is exported to ExecuTorch `.pte` format for efficient inference on Arm-based Edge Linux systems using the XNNPACK backend.

## Summary

This repository contains an Arm-optimized version of yolo11n for object detection. The model is provided in ExecuTorch `.pte` format, targeting Edge Linux systems.

This version is intended to demonstrate efficient inference on Arm-based platforms while preserving the original model's intended behavior. Arm has evaluated this model on COCO 2017 (val2017) and measured performance on a representative evaluation target.

**Key results**

| Area | Result |
|---|---|
| Model format | ExecuTorch `.pte` |
| Target device class | Edge Linux |
| Reference device | Raspberry Pi 5 (Cortex-A76, Linux, Raspberry Pi OS 64-bit, based on Debian 13 "Trixie") |
| Primary performance result | 7.46 FPS (134.10 ms p50 latency) |
| Accuracy result | mAP@0.5:0.95 38.90% |
| Size / memory result | 4.73 MB, 2.16x smaller than the FP32 baseline |

## Original model

| Field | Value |
|---|---|
| Original model | yolo11n |
| Original source | GitHub |
| Original developer | Ultralytics |
| Original model card | https://huggingface.co/Ultralytics/YOLO11 |
| Original license | AGPL-3.0 (a commercial Enterprise license is also available from Ultralytics for use cases that cannot comply with AGPL-3.0) |

## Model files

| File | Description |
|---|---|
| `yolo11n_raspberry_executorch_optimized.pte` | Arm-optimized model for deployment |
| `example.py` | Minimal inference example |
| `pyproject.toml` | Pinned runtime dependencies and deployment metadata |
| `uv.lock` | Locked dependency versions, sources, and hashes for `uv sync --frozen` |
| `config.yaml` | Model I/O contract used by the example |
| `benchmarks/` | FP32 baseline and Arm-optimized benchmark records |

## Performance

Performance was measured on the reference configuration below. Results are intended to make the optimization reproducible but do not guarantee identical performance on every Arm-based system.

**Reference configuration**

| Field | Value |
|---|---|
| Device / platform | Raspberry Pi 5 |
| CPU / accelerator | Cortex-A76 (4 cores @ 2.4 GHz), CPU |
| OS | Linux, Raspberry Pi OS 64-bit, based on Debian 13 "Trixie" |
| Runtime | ExecuTorch 1.1.0 |
| Backend / delegate | XNNPACK, KleidiAI |
| Batch size | 1 |
| Precision | Mixed precision: INT8 for eligible backbone and neck operations; the stem convolution (model.0.conv) and complete detection head (model.23) remain in FP32 |
| Quantization method | Static PTQ with symmetric, per-channel weight quantization |
| Runs | 10 warmup + 100 measured |

**Performance results**

| Metric | Original / baseline | Arm-optimized | Improvement |
|---|---:|---:|---:|
| p50 latency | 169.36 ms | 134.10 ms | 1.26x faster |
| p90 latency | 170.12 ms | 134.68 ms | 1.26x faster |
| p99 latency | 171.62 ms | 135.06 ms | 1.27x faster |
| Frames per second | 5.90 | 7.46 | 1.26x higher |
| Model size | 10.23 MB | 4.73 MB | 2.16x smaller |
| Model load time | 12.84 ms | 8.82 ms | 1.46x faster |
| Time to first inference | 191.71 ms | 150.83 ms | 1.27x faster |
| Peak memory | 78.33 MB | 59.17 MB | 1.32x less |
| Average memory | 77.42 MB | 57.98 MB | 1.34x less |

## Accuracy

Accuracy was evaluated using the same preprocessing, input resolution, and evaluation protocol described below. Where possible, the optimized model is compared against the original model under the same evaluation conditions.

**Evaluation setup**

| Field | Value |
|---|---|
| Dataset | COCO 2017 |
| Split | val2017 |
| Number of samples | 5,000 |
| Metric(s) | mAP@0.5:0.95, mAP@0.5, mAP@0.75 |
| Evaluation runtime | ExecuTorch (Arm-optimized) / PyTorch (FP32 baseline) |

**Accuracy results**

| Metric | Original / baseline | Arm-optimized | Change |
|---|---:|---:|---:|
| mAP@0.5:0.95 | 39.30% | 38.90% | -0.40 pp |
| mAP@0.5 | 55.11% | 54.66% | -0.45 pp |
| mAP@0.75 | 42.75% | 42.38% | -0.37 pp |

Accuracy was measured using the evaluation setup described above. Users should re-evaluate the model on their own data before production use.

## Arm optimization approach

Arm optimized this model for efficient inference on Arm-based platforms using a hardware-aware conversion and validation flow.

For this release, Arm used:

| Optimization area | Applied? | Notes |
|---|---|---|
| Model conversion | Yes | Converted to ExecuTorch `.pte`, targeting the XNNPACK backend |
| Quantization | Yes | Static INT8 PTQ with symmetric, per-channel weight quantization, calibrated on 2,000 COCO 2017 training images. The stem convolution and complete detection head (model.23) remain in FP32; remaining concatenations and their direct consumers are excluded from the standard quantization path |
| Runtime/backend selection | Yes | XNNPACK backend with KleidiAI kernels, 1 thread |
| Graph/runtime compatibility updates | Yes | Internal C3k2, C3k, SPPF, and C2PSA concatenations are rewritten before capture as independent per-branch convolutions with folded BatchNorm. After conversion, forced Q/DQ pairs are removed from remaining concatenations network-wide, including three detection-head concatenations and the final output merge, located through FX graph topology |
| Accuracy validation | Yes | Compared against the original model or published baseline |
| Performance validation | Yes | Measured on the reference Arm platform |

The goal of this process is to improve deployment characteristics such as latency, memory use, model size, and runtime compatibility while preserving the model's intended behavior. Detailed conversion scripts, calibration configuration, or backend-specific implementation details may be provided separately where appropriate.

## Using this model

### Install dependencies

```bash
uv python install
uv sync --frozen
```

### Run the example

```bash
uv run example.py
```

### Expected input

| Property | Value |
|---|---|
| Input shape | [1, 3, 640, 640] |
| Input type | float32 |
| Input range | [0.0, 1.0] |
| Preprocessing | Letterbox-resize to 640x640 (aspect-ratio-preserving resize + pad, pad value 114/255, center-anchored); convert to tensor by resizing uint8 pixels first and then dividing by 255 to obtain [0, 1] float32 values (Ultralytics order); no mean/std normalization |

### Expected output

| Property | Value |
|---|---|
| Output shape | [1, 84, 8400] |
| Output type | N/A |
| Postprocessing | Transpose to [1, 8400, 84] (4 xywh center coordinates + 80 class scores, no objectness); score = max(class_scores); filter below the confidence threshold (0.001 for the reported mAP, so the full precision-recall curve is retained; the bundled example uses 0.4 for readable output); convert xywh to xyxy; keep top 30,000 candidates by score; per-class batched NMS (IoU threshold 0.7); keep top 300 detections; rescale via the inverse letterbox transform back to original image coordinates |

## Intended use

This model is intended for developers evaluating object detection workloads on Arm-based platforms. It is suitable as a reference implementation for benchmarking, prototyping, and integration exploration.

## Limitations

- Performance depends on the target device, runtime version, backend/delegate support, memory configuration, and system load.
- Accuracy was evaluated on COCO 2017 val2017 and may not generalize to all domains.
- This release preserves the original model's intended task and behavior, but users should validate it for their own application, data, and deployment environment.
- This repository is not a replacement for the original model documentation.

## Additional notes

- Fixed input size of 640x640 with letterbox padding; dynamic or arbitrary input resolutions require re-export.
- Sample input: `sample_input.jpg` is derived from [Living room (Unsplash)](https://commons.wikimedia.org/wiki/File:Living_room_(Unsplash).jpg) by Jarosław Ceborski, via Wikimedia Commons ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)).

## About this version

Original Model:
yolo11n by Ultralytics - [Repository](https://huggingface.co/Ultralytics/YOLO11)

Optimization/conversion:
Arm-Optimized version for execution on Arm-based platforms.

Converted/optimized by:
Arm

License:
The Original Model and the Optimized Model are subject to [AGPL-3.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/agpl-3.0.md) (a commercial Enterprise license is also available from Ultralytics for use cases that cannot comply with AGPL-3.0).

This repository contains a converted or optimized version of the Original Model (the “Optimized Model”). The Original Model has been converted or optimized as described above for execution on Arm-based platforms.

No retraining or fine-tuning of the Original Model was performed as part of the conversion or optimization. The conversion or optimization was not intended to change the Original Model’s behavior or intended use.

### Original Model and Documentation

For information about the Original Model, including its development, training data, intended uses, limitations and other relevant information, please refer to the [Original Model repository](https://huggingface.co/Ultralytics/YOLO11). Information in that repository was provided by the original developer or other third parties and, unless expressly stated otherwise, has not been independently verified by Arm.

### Licenses and Third-Party Terms

Use of the Original Model and the Optimized Model is subject to the applicable licenses, usage restrictions and other terms identified above and in the relevant repositories. Publication of the Optimized Model does not grant any rights beyond those provided under the applicable license terms.

You are responsible for reviewing those terms and ensuring that your use of the Original Model and the Optimized Model is permitted.

### Purpose of this Release

The Optimized Model is provided as a reference implementation to demonstrate and evaluate execution and performance on Arm-based systems. It is not a production-ready or supported solution.

Arm’s publication of the Optimized Model does not constitute an endorsement or certification of the Original Model or a representation that the Optimized Model is suitable for production use or any particular purpose.

To the fullest extent permitted by applicable law (i) the Optimized Model is provided “as is.” Arm makes no representations or warranties that the Original Model, the Optimized Model or their outputs are accurate, safe, secure, non-infringing, legally compliant, suitable for production use or fit for any particular purpose; and (ii) Arm will not be liable for any loss or damage arising from or in connection with the Optimized Model, its use or its outputs.

You are responsible for independently evaluating the Optimized Model, its outputs and its suitability for your intended use, including compliance with applicable legal, regulatory, safety and security requirements.

Arm does not commit to provide ongoing support, maintenance or updates for the Optimized Model. Any use of or reliance on the Optimized Model or its outputs is at your own risk.

## Use of Ultralytics models

Ultralytics provides Ultralytics YOLO software and models available through the Arm AI Portal and/or Arm's Hugging Face Organization under the GNU Affero General Public License v3.0 ("AGPL-3.0"), unless you have entered into a separate written license agreement with Ultralytics. Accessing, downloading, or retraining these materials through Arm AI Portal does not grant you an Ultralytics Enterprise License or any other proprietary Ultralytics license.

AGPL-3.0 requires you to release the complete source code of any application that uses Ultralytics YOLO, including applications made available over a network, under the same license. If you are embedding YOLO in a commercial product, internal tool, or production deployment and cannot open-source your code, you need an Ultralytics Enterprise License.

You are responsible for determining which applies to your use. Terms and enterprise licensing options are available at [ultralytics.com](https://www.ultralytics.com/license?utm_source=arm&utm_medium=referral&utm_content=license_notice_message).
