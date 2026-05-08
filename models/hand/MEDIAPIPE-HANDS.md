# Documentation

## MediaPipe Hands

Source: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

### Overview

MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame. It achieves real-time performance on a mobile phone and scales to multiple hands.

### ML Pipeline

Two models working together:

1. **Palm detection model**: operates on the full image, returns an oriented hand bounding box
2. **Hand landmark model**: operates on the cropped image region defined by the palm detector, returns high-fidelity 3D hand keypoints

The accurately cropped hand image drastically reduces the need for data augmentation (rotations, translation, scale) and allows the network to dedicate most of its capacity towards coordinate prediction accuracy.

Crops can be generated based on hand landmarks from the previous frame. Only when the landmark model can no longer identify hand presence is palm detection re-invoked.

### Palm Detection Model

- Single-shot detector optimized for mobile real-time
- Detects palms instead of hands (rigid objects = simpler bounding boxes)
- Square bounding boxes (anchors), reducing anchor count by 3-5x
- Encoder-decoder feature extractor for scene context awareness
- Focal loss for large anchor count from high scale variance
- Average precision: 95.7% (vs 86.22% baseline with cross entropy + no decoder)
- Handles ~20x scale span relative to image frame
- Non-maximum suppression works well even for two-hand self-occlusion (handshakes)

### Hand Landmark Model

- Precise keypoint localization of 21 3D hand-knuckle coordinates via regression (direct coordinate prediction)
- Learns a consistent internal hand pose representation
- Robust to partially visible hands and self-occlusions
- Training data: ~30K manually annotated real-world images with 21 3D coordinates (z from depth maps where available) + rendered synthetic hand models over various backgrounds

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `static_image_mode` | If false, treats input as video stream and tracks between frames. If true, runs detection on every frame. | false |
| `max_num_hands` | Maximum number of hands to detect. | 2 |
| `model_complexity` | 0 or 1. Landmark accuracy and inference latency go up with complexity. | 1 |
| `min_detection_confidence` | [0.0, 1.0] Minimum confidence from palm detection model. | 0.5 |
| `min_tracking_confidence` | [0.0, 1.0] Minimum confidence from landmark-tracking model for landmarks to be considered tracked. Otherwise hand detection is re-invoked. Higher = more robust, higher latency. Ignored if `static_image_mode` is true. | 0.5 |

### Outputs

#### `multi_hand_landmarks`

Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks and each landmark is composed of x, y and z.

- x and y are normalized to [0.0, 1.0] by the image width and height respectively
- z represents the landmark depth with the depth at the wrist being the origin
- The smaller the z value the closer the landmark is to the camera
- **The magnitude of z uses roughly the same scale as x**

#### `multi_hand_world_landmarks`

Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks in world coordinates.

- Each landmark is composed of x, y and z: **real-world 3D coordinates in meters**
- Origin at the hand's approximate geometric center

#### `multi_handedness`

Collection of handedness of the detected/tracked hands (left or right).

- Each hand is composed of label and score
- label: "Left" or "Right"
- score: estimated probability of predicted handedness, always >= 0.5
- **Handedness is determined assuming the input image is mirrored** (front-facing/selfie camera with images flipped horizontally). If not the case, swap the handedness output.

### Landmark Indices

```
 0: WRIST
 1: THUMB_CMC    5: INDEX_MCP    9: MIDDLE_MCP   13: RING_MCP    17: PINKY_MCP
 2: THUMB_MCP    6: INDEX_PIP   10: MIDDLE_PIP   14: RING_PIP    18: PINKY_PIP
 3: THUMB_IP     7: INDEX_DIP   11: MIDDLE_DIP   15: RING_DIP    19: PINKY_DIP
 4: THUMB_TIP    8: INDEX_TIP   12: MIDDLE_TIP   16: RING_TIP    20: PINKY_TIP
```

### Hand Connections (bones)

```
WRIST -> THUMB_CMC -> THUMB_MCP -> THUMB_IP -> THUMB_TIP
WRIST -> INDEX_MCP -> INDEX_PIP -> INDEX_DIP -> INDEX_TIP
WRIST -> MIDDLE_MCP -> MIDDLE_PIP -> MIDDLE_DIP -> MIDDLE_TIP
WRIST -> RING_MCP -> RING_PIP -> RING_DIP -> RING_TIP
WRIST -> PINKY_MCP -> PINKY_PIP -> PINKY_DIP -> PINKY_TIP
INDEX_MCP -> MIDDLE_MCP -> RING_MCP -> PINKY_MCP (palm base)
```
