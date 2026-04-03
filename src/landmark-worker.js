// Landmark inference worker: loads its own ONNX session, runs inference on demand.
// Each worker gets its own WASM instance and WebGPU device -- true parallel.

import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

let session = null;
let outputMap = null;

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });

      // Warmup + discover output names
      const warmup = new ort.Tensor('float32', new Float32Array(224 * 224 * 3), [1, 224, 224, 3]);
      const results = await session.run({ [session.inputNames[0]]: warmup });

      outputMap = {};
      for (const name of session.outputNames) {
        const t = results[name];
        const size = t.data.length;
        if (size === 63 && !outputMap.landmarks) outputMap.landmarks = name;
        else if (size === 63) outputMap.worldLandmarks = name;
        else if (size === 1 && !outputMap.handFlag) outputMap.handFlag = name;
        else if (size === 1) outputMap.handedness = name;
      }

      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    try {
      const inputData = new Float32Array(e.data.inputBuffer);
      const tensor = new ort.Tensor('float32', inputData, [1, 224, 224, 3]);

      const results = await session.run({ [session.inputNames[0]]: tensor });

      const landmarks = outputMap.landmarks ? new Float32Array(results[outputMap.landmarks].data) : new Float32Array(0);
      const handFlag = outputMap.handFlag ? results[outputMap.handFlag].data[0] : 0;
      const handedness = outputMap.handedness ? results[outputMap.handedness].data[0] : 0;

      self.postMessage({
        type: 'result',
        id: e.data.id,
        landmarksBuffer: landmarks.buffer,
        handFlag,
        handedness,
      }, [landmarks.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', id: e.data.id, message: err.message });
    }
  }
};
