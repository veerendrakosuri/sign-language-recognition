// Browser version of signlang.features.prepare and the scikit-learn baseline.
// Keep this module dependency-free so parity can be checked with Node.

export function assignFrame(poseLandmarks = [], handLandmarks = []) {
  const points = Array.from({ length: 75 }, () => [0, 0, 0]);
  const valid = Array(75).fill(false);
  if (poseLandmarks.length) {
    poseLandmarks[0].forEach((point, i) => {
      if (i < 33 && (point.visibility ?? 1) >= 0.5 && (point.presence ?? 1) >= 0.5) {
        points[i] = [point.x, point.y, point.z];
        valid[i] = true;
      }
    });
  }
  const wrists = [15, 16].filter((i) => valid[i]);
  const hands = handLandmarks.slice(0, 2);
  const choices = [];
  const distance = (h, w) => Math.hypot(hands[h][0].x - points[w][0], hands[h][0].y - points[w][1]);
  for (let h = 0; h < hands.length; h += 1) {
    for (const w of wrists) {
      const d = distance(h, w);
      if (d <= 0.25) choices.push({ assignments: [[h, w]], cost: d });
    }
  }
  if (hands.length >= 2 && wrists.length >= 2) {
    for (const pair of [[[0, 15], [1, 16]], [[0, 16], [1, 15]]]) {
      const ds = pair.map(([h, w]) => distance(h, w));
      if (Math.max(...ds) <= 0.25) choices.push({ assignments: pair, cost: ds[0] + ds[1] });
    }
  }
  const maxCount = Math.max(0, ...choices.map((c) => c.assignments.length));
  const best = choices.filter((c) => c.assignments.length === maxCount).sort((a, b) => a.cost - b.cost)[0];
  for (const [h, wrist] of best?.assignments ?? []) {
    const start = wrist === 15 ? 33 : 54;
    hands[h].forEach((point, i) => {
      if (i < 21) {
        points[start + i] = [point.x, point.y, point.z];
        valid[start + i] = true;
      }
    });
  }
  return { points, valid };
}

function interpolatedReference(frames, good, index, coordinate) {
  let left = -1;
  let right = -1;
  for (let j = index; j >= 0; j -= 1) if (good[j]) { left = j; break; }
  for (let j = index; j < frames.length; j += 1) if (good[j]) { right = j; break; }
  if (left < 0) left = right;
  if (right < 0) right = left;
  const center = (j) => (frames[j].points[11][coordinate] + frames[j].points[12][coordinate]) / 2;
  if (left === right) return center(left);
  const fraction = (index - left) / (right - left);
  return center(left) * (1 - fraction) + center(right) * fraction;
}

export function prepareMean(frames, outputFrames = 32, summary = 'zero_mean') {
  if (!frames.length || outputFrames < 2) throw new Error("A non-empty sign video is required.");
  if (!['zero_mean', 'observed_mean'].includes(summary)) throw new Error('Invalid model summary.');
  const widths = frames.map(({ points }) => Math.hypot(points[11][0] - points[12][0], points[11][1] - points[12][1]));
  const good = frames.map(({ valid }, i) => valid[11] && valid[12] && widths[i] > 1e-4);
  const usableWidths = widths.filter((_, i) => good[i]).sort((a, b) => a - b);
  if (!usableWidths.length) throw new Error("Keep both shoulders visible throughout the sign.");
  const middle = Math.floor(usableWidths.length / 2);
  const scale = usableWidths.length % 2 ? usableWidths[middle] : (usableWidths[middle - 1] + usableWidths[middle]) / 2;
  const normalized = frames.map(({ points, valid }, i) => {
    const center = [0, 1, 2].map((d) => interpolatedReference(frames, good, i, d));
    return points.flatMap((point, landmark) => {
      if (!valid[landmark]) return [0, 0, 0];
      return [(point[0] - center[0]) / scale,
        (point[1] - center[1]) / scale,
        (point[2] - (landmark < 33 ? center[2] : 0)) / scale];
    });
  });
  const mean = Array(225).fill(0);
  const counts = Array(225).fill(0);
  for (let i = 0; i < outputFrames; i += 1) {
    // numpy.round uses ties-to-even; 32-frame clips almost never hit a tie.
    const index = Math.round(i * (normalized.length - 1) / (outputFrames - 1));
    normalized[index].forEach((value, d) => {
      const observed = frames[index].valid[Math.floor(d / 3)];
      if (summary === 'zero_mean') mean[d] += value / outputFrames;
      else if (observed) { mean[d] += value; counts[d] += 1; }
    });
  }
  if (summary === 'observed_mean') mean.forEach((value, d) => { mean[d] = counts[d] ? value / counts[d] : 0; });
  return mean;
}

export function classify(frames, model) {
  if (!model || model.schema !== 1 || model.mean?.length !== 225 || model.scale?.length !== 225 ||
      model.weights?.length !== model.labels?.length || model.bias?.length !== model.labels?.length) {
    throw new Error("The recognition model is unavailable or invalid.");
  }
  if (model.summary && !['zero_mean', 'observed_mean'].includes(model.summary)) {
    throw new Error("The recognition model is unavailable or invalid.");
  }
  const mean = prepareMean(frames, model.frames, model.summary ?? 'zero_mean');
  const scaled = mean.map((x, i) => (x - model.mean[i]) / model.scale[i]);
  const logits = model.weights.map((weights, classIndex) => {
    if (weights.length !== 225) throw new Error("Invalid model weights.");
    return weights.reduce((sum, weight, i) => sum + weight * scaled[i], model.bias[classIndex]);
  });
  const max = Math.max(...logits);
  const exponentials = logits.map((x) => Math.exp(x - max));
  const denominator = exponentials.reduce((sum, x) => sum + x, 0);
  return model.labels.map((label, i) => ({ label, score: exponentials[i] / denominator }))
    .sort((a, b) => b.score - a.score);
}
