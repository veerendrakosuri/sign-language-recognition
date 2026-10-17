import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { assignFrame, classify, prepareMean } from '../inference.mjs';

const model = JSON.parse(readFileSync(new URL('../public/model.json', import.meta.url), 'utf8'));

function syntheticFrames() {
  return Array.from({ length: 5 }, (_, t) => {
    const points = Array.from({ length: 75 }, (_, i) => [
      0.5 + (i % 5) * 0.01 + t * 0.002,
      0.2 + Math.floor(i / 5) * 0.01,
      i * 0.01,
    ]);
    points[11] = [0.35 + t * 0.002, 0.45, 0.1];
    points[12] = [0.65 + t * 0.002, 0.45, 0.15];
    for (let i = 33; i < 75; i += 1) points[i] = [
      0.4 + ((i - 33) % 21) * 0.005 + t * 0.003,
      0.5 + ((i - 33) % 21) * 0.004,
      i * 0.001,
    ];
    const valid = Array(75).fill(true);
    if (t === 1 || t === 3) valid[5] = false;
    if (t === 2) valid.fill(false, 54);
    return { points, valid };
  });
}

test('browser preprocessing agrees with Python on synthetic landmark frames', () => {
  const mean = prepareMean(syntheticFrames());
  const pythonStart = [-4.9670543234014986e-8, -0.8333334922790527, -0.41666674613952637,
    0.03333330899477005, -0.8333334922790527, -0.3833332657814026];
  const pythonEnd = [-0.00750000262632966, 0.3150000274181366, 0.18250000476837158,
    0.005000011064112186, 0.32500001788139343, 0.18500006198883057];
  for (let i = 0; i < pythonStart.length; i += 1) assert.ok(Math.abs(mean[i] - pythonStart[i]) < 1e-5);
  for (let i = 0; i < pythonEnd.length; i += 1) assert.ok(Math.abs(mean[mean.length - pythonEnd.length + i] - pythonEnd[i]) < 1e-5);
  const predictions = classify(syntheticFrames(), model);
  assert.equal(predictions.length, 100);
  assert.equal(predictions[0].label, 'thursday');
  assert.ok(Math.abs(predictions.reduce((sum, row) => sum + row.score, 0) - 1) < 1e-10);
});

test('observed mean does not average missing landmarks as zeros', () => {
  const frames = syntheticFrames();
  const zeroMean = prepareMean(frames, 5, 'zero_mean');
  const observedMean = prepareMean(frames, 5, 'observed_mean');
  const rightHandCoordinate = 54 * 3;
  assert.notEqual(zeroMean[rightHandCoordinate], observedMean[rightHandCoordinate]);
  assert.ok(Math.abs(observedMean[rightHandCoordinate] -
    (zeroMean[rightHandCoordinate] * 5 / 4)) < 1e-6);
});

test('hand slots follow anatomical pose wrists rather than handedness labels', () => {
  const pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1, presence: 1 }));
  pose[15].x = 0.2; pose[16].x = 0.8;
  const right = Array.from({ length: 21 }, () => ({ x: 0.81, y: 0, z: 0.02 }));
  const left = Array.from({ length: 21 }, () => ({ x: 0.19, y: 0, z: 0.03 }));
  const frame = assignFrame([pose], [right, left]);
  assert.equal(frame.points[33][0], 0.19);
  assert.equal(frame.points[54][0], 0.81);
  assert.ok(frame.valid[33] && frame.valid[54]);
});

test('missing shoulders reject classification', () => {
  const frames = syntheticFrames();
  frames.forEach((frame) => { frame.valid[11] = false; frame.valid[12] = false; });
  assert.throws(() => classify(frames, model), /shoulders/);
});
