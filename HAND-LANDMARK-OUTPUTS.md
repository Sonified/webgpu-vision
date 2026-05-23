# Hand Landmark Model Outputs

Reference: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

## Two Landmark Outputs (both 21x3 = 63 floats)

The hand landmark model produces TWO sets of landmarks, not one:

### 1. Normalized Landmarks (`multi_hand_landmarks`)

- x, y normalized to [0.0, 1.0] by image width/height
- z is depth relative to wrist (wrist z = 0, smaller = closer to camera)
- **"The magnitude of z uses roughly the same scale as x"** (per MediaPipe docs)
- This is what we currently use (`outputNames.landmarks`)
- These go through inverse affine projection before reaching the main thread
- After projection: x, y are divided by viewport width/height; z is divided by the affine scale factor S

### 2. World Landmarks (`multi_hand_world_landmarks`)

- Real-world 3D coordinates **in meters**
- Origin at the hand's approximate geometric center
- No projection, no image-space normalization
- True 3D positions -- orientation is just a cross product away
- **We discover this output but never use it** (`outputNames.worldLandmarks` in `landmark-worker-wgsl.js:144`)

## The Necker Cube Problem

When a 3D hand is projected onto a 2D screen, the viewer perceives an optical illusion: the same 2D silhouette can be interpreted as either "palm facing toward me" (model's reality) or "back of hand reaching into the screen" (what the user actually perceives). This is identical to the spinning dancer illusion or the Necker cube.

For game/AR use cases (Spellaria, ball-toss), the user wants the "back of hand into the scene" interpretation. This means the z-axis of the orientation frame needs to be flipped relative to what the model reports. With normalized landmarks, this creates cascading coordinate system confusion. With world landmarks, it should be a single negation of the normal vector after computing orientation in clean 3D space.

## What We Tried With Normalized Landmarks (and why it failed)

### Approach 1: Scene-space cross product
Computed `(wrist->index_MCP) x (wrist->pinky_MCP)` using `landmarkToScene()` output.

Problem: `landmarkToScene` applies non-uniform scaling (x*-12, y*-9, z*-3). The z contribution is 4x weaker than x, so depth barely affects the cross product. Palm flat to camera gave yaw=90 (sideways!) instead of ~0 or ~180.

### Approach 2: Landmark-space cross product with z scaling
Computed cross product in raw landmark coordinates with `HAND_Z_SCALE = 3.7` (empirically derived from palm rotation data), then transformed to scene space via the Jacobian.

Problem: The Jacobian (-12, -9, -3) negates all axes, flipping cross product handedness. Combined with the Necker z-flip, the double negation cancelled out in confusing ways. Rotation tracked in one direction but not the other, and the normal whipped around at knife-edge orientations.

### Approach 3: Spellaria's `getPalmNormal`
Reference: `explorations/spellaria/grid-3d.html:3776`

```javascript
const v1 = new THREE.Vector3().subVectors(idx, w);
const v2 = new THREE.Vector3().subVectors(pnk, w);
const n = new THREE.Vector3().crossVectors(v1, v2).normalize();
if (n.z < 0) n.negate();  // force forward
```

This worked for Spellaria's use case (projectiles aimed roughly forward) but the `if (n.z < 0) negate` hack causes flip-flopping at knife-edge angles. Not suitable for continuous orientation tracking.

### Empirical z-scale data

Hand held upright, rotated from palm-facing to knife-edge:
- Palm facing camera: `2d=0.245, pz_raw=-0.002` (nearly no z difference)
- Knife edge: `2d=0.174, pz_raw=-0.047` (significant z difference)
- Ratio: the 2D distance drops ~29% while z grows ~24x
- Derived `HAND_Z_SCALE = 3.7` to bring z into same units as x/y

This scale factor contradicts the MediaPipe docs claiming "z uses roughly the same scale as x." Possible explanations:
- The docs describe the raw model output; our z goes through `oz / S` (affine scale division) which changes its units
- The ONNX model we use may have different output scaling than MediaPipe's internal pipeline
- "Roughly" may be doing a lot of work in that sentence

### The 3D bone length test

Even with z scaled by 3.7, the 3D distance wrist(0)->middle_MCP(9) swung from 0.20 to 0.35 during rotation (1.75x ratio). This should be nearly constant for a rigid bone. Conclusion: normalized landmark z is not accurate enough for rotation-invariant measurements.

## Why World Landmarks Solve This

World landmarks are in meters, in true 3D, with the origin at the hand's geometric center. They bypass every problem we hit:
- No image-space projection artifacts
- No non-uniform scaling from `landmarkToScene`
- No ambiguous z units or scale factors
- Bone lengths should be genuinely constant
- Cross product of palm triangle edges gives clean orientation
- The only remaining step is the Necker flip (negate normal for "into screen" interpretation)

## CRITICAL DISCOVERY: World Landmarks Are Hand-Relative (2024-04-24)

World landmarks are in a coordinate frame that **rotates with the hand**. The "world" means metric scale, NOT camera-relative orientation.

### Evidence

Recorded a user rotating their hand from palm-facing-camera through knife-edge and back. The cross product normal of the palm triangle (wrist/index_MCP/pinky_MCP) in world coordinates:

- `nx` stayed between 0.82 and 0.99 across ALL orientations
- The normal barely changed despite dramatic hand rotation
- This is because the palm triangle's relative geometry is roughly constant (rigid bones)

The wrist, index_MCP, and pinky_MCP positions DO shift in world coords as the hand rotates, but the axes rotate WITH the hand, so the cross product stays roughly the same.

### What World Landmarks ARE Good For

1. **Rotation-invariant palm size for Z estimation.** Palm width is physically ~6.3cm regardless of orientation. Combine with normalized landmark screen-space width (which shrinks with distance) for absolute depth via similar triangles. No more first-frame calibration needed.
2. **Finger pose / gesture detection.** Bone angles relative to each other (finger curl) without camera perspective distortion.
3. **Physical hand measurements.** True bone lengths in meters.

### What World Landmarks CANNOT Do

- **Palm orientation relative to camera.** The coordinate frame spins with the hand so the cross product is useless for determining which way the palm faces.
- **Absolute position in space.** Origin is at the hand's geometric center, not camera origin.

### Implications for Orientation

Orientation MUST come from normalized landmarks (which ARE in camera/image space). The challenge remains: normalized landmark z is weak and in mystery units after `oz / S` affine scale division. The `HAND_Z_SCALE = 3.7` empirical correction partially compensated but bone lengths still swung 1.75x during rotation.

Possible approaches still to explore:
- Compare world landmark bone directions to normalized landmark bone directions; the rotation between them IS the hand orientation
- Use normalized landmarks more carefully with better z calibration
- Use the ratio of foreshortened vs true bone lengths to infer tilt angles

## Action Items

1. ~~Wire `worldLandmarks` through `landmark-worker-wgsl.js` to the main thread~~ DONE
2. Use world landmarks for:
   - ~~Palm orientation~~ NOT POSSIBLE (hand-relative frame)
   - Hand Z depth estimation (true 3D palm size is rotation-invariant)
   - Finger pose / gesture detection
3. Keep normalized landmarks for 2D screen-space positioning AND orientation
4. Orientation still unsolved -- needs a fresh approach using normalized landmarks or world-vs-normalized comparison

## Other Useful Model Parameters

From the MediaPipe docs:

- **model_complexity**: 0 or 1. Higher = better landmark accuracy + higher latency. Default 1. We should verify which complexity our ONNX model was exported with.
- **min_detection_confidence**: [0.0, 1.0]. Threshold for palm detector. Default 0.5.
- **min_tracking_confidence**: [0.0, 1.0]. Below this, re-runs palm detection instead of tracking. Higher = more robust but higher latency. Default 0.5. We implement this via `handFlag` threshold.

## `landmarkToScene` Reference

Ball-toss (`demos/ball-toss/index.html`):
```javascript
function landmarkToScene(lm) {
  const handZ = camera.position.z - (6 + headPos.z * -4) + lm.z * -3;
  const pFrac = camera.position.z !== 0 ? handZ / camera.position.z : 0;
  return new THREE.Vector3(
    -(lm.x - 0.5) * 12 + camera.position.x * pFrac,  // x: negated, scale 12
    -(lm.y - 0.5) * 9  + camera.position.y * pFrac,   // y: negated, scale 9
    handZ                                               // z: scale 3, negated
  );
}
```

Spellaria (`explorations/spellaria/grid-3d.html`):
```javascript
function landmarkToScene(lm) {
  const handZ = camera.position.z + 8 + headPos.z * 3 + lm.z * -4;
  return new THREE.Vector3(
    (lm.x - 0.5) * 16 + camera.position.x,   // x: NOT negated, scale 16
    -(lm.y - 0.5) * 12 + camera.position.y,   // y: negated, scale 12
    handZ                                       // z: scale 4, negated
  );
}
```

Key difference: ball-toss negates x (mirror), Spellaria does not. This flips the handedness of cross products computed in scene space.

## Landmark Indices

```
 0: WRIST
 1: THUMB_CMC    5: INDEX_MCP    9: MIDDLE_MCP   13: RING_MCP    17: PINKY_MCP
 2: THUMB_MCP    6: INDEX_PIP   10: MIDDLE_PIP   14: RING_PIP    18: PINKY_PIP
 3: THUMB_IP     7: INDEX_DIP   11: MIDDLE_DIP   15: RING_DIP    19: PINKY_DIP
 4: THUMB_TIP    8: INDEX_TIP   12: MIDDLE_TIP   16: RING_TIP    20: PINKY_TIP
```

Palm orientation triangle: wrist(0), index_MCP(5), pinky_MCP(17).
