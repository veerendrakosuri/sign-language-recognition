import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { assignFrame, classify } from './inference.mjs';

const VERSION = '0.10.35';
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const POSE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const HAND = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const input = document.querySelector('#video-file');
const dropzone = document.querySelector('#dropzone');
const video = document.querySelector('#preview');
const previewWrap = document.querySelector('#preview-wrap');
const name = document.querySelector('#file-name');
const button = document.querySelector('#analyze');
const status = document.querySelector('#status');
const progressWrap = document.querySelector('#progress-wrap');
const progressBar = document.querySelector('#progress-bar');
let objectUrl;
let busy = false;
let modelPromise;

function say(message, error = false) {
  status.textContent = message;
  status.classList.toggle('error', error);
}

function once(target, event, timeout = 15000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error(`Timed out waiting for ${event}.`)); }, timeout);
    function cleanup() { clearTimeout(timer); target.removeEventListener(event, done); target.removeEventListener('error', failed); }
    function done() { cleanup(); resolve(); }
    function failed() { cleanup(); reject(new Error('Your browser could not decode this video. Try an MP4 (H.264) or WebM clip.')); }
    target.addEventListener(event, done, { once: true });
    target.addEventListener('error', failed, { once: true });
  });
}

async function setFile(file) {
  if (busy || !file) return;
  if (!file.type.startsWith('video/')) { say('Choose an MP4 or WebM video file.', true); return; }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  video.load();
  name.textContent = file.name;
  previewWrap.hidden = false;
  button.disabled = true;
  document.querySelector('#result').hidden = true;
  document.querySelector('#empty-result').hidden = false;
  try {
    if (video.readyState < 1) await once(video, 'loadedmetadata');
    if (!Number.isFinite(video.duration) || video.duration < 0.5 || video.duration > 30) {
      throw new Error('Choose a trimmed clip between 0.5 and 30 seconds.');
    }
    button.disabled = false;
    say(`Ready: ${video.duration.toFixed(1)} seconds. Press Analyze sign.`);
  } catch (error) { say(error.message, true); }
}

input.addEventListener('change', () => setFile(input.files?.[0]));
dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (event) => { event.preventDefault(); dropzone.classList.remove('dragover'); setFile(event.dataTransfer.files?.[0]); });
window.addEventListener('beforeunload', () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });

async function seek(time) {
  const event = once(video, 'seeked');
  video.currentTime = time;
  await event;
}

function showResults(predictions) {
  const first = predictions[0];
  document.querySelector('#top-label').textContent = first.label;
  document.querySelector('#top-score').textContent = `${Math.round(first.score * 100)}% model score`;
  const alternatives = document.querySelector('#alternatives');
  alternatives.replaceChildren();
  for (const item of predictions.slice(1, 5)) {
    const row = document.createElement('div'); row.className = 'alternative';
    const label = document.createElement('span'); label.textContent = item.label;
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.width = `${Math.max(1, item.score * 100)}%`;
    const score = document.createElement('output'); score.textContent = `${Math.round(item.score * 100)}%`;
    track.append(fill); row.append(label, track, score); alternatives.append(row);
  }
  document.querySelector('#empty-result').hidden = true;
  document.querySelector('#result').hidden = false;
}

async function loadModel() {
  const response = await fetch('./model.json');
  if (!response.ok) throw new Error('The trained model could not be downloaded. Try reloading the page.');
  return response.json();
}

async function analyze() {
  if (busy || button.disabled) return;
  busy = true; button.disabled = true; video.pause();
  document.querySelector('#result').hidden = true;
  document.querySelector('#empty-result').hidden = false;
  progressWrap.hidden = false; progressBar.style.width = '0%';
  let pose;
  let hands;
  try {
    say('Loading recognition models…');
    modelPromise ??= loadModel();
    const model = await modelPromise;
    const fileset = await FilesetResolver.forVisionTasks(WASM);
    pose = await PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: POSE, delegate: 'CPU' }, runningMode: 'VIDEO', numPoses: 1 });
    hands = await HandLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: HAND, delegate: 'CPU' }, runningMode: 'VIDEO', numHands: 2 });
    const frames = [];
    // 32 monotonic timestamps cover the complete clip while avoiding the blank last frame.
    for (let i = 0; i < model.frames; i += 1) {
      const time = 0.01 + (video.duration - 0.03) * i / (model.frames - 1);
      await seek(time);
      const timestamp = Math.round(time * 1000);
      const p = pose.detectForVideo(video, timestamp);
      const h = hands.detectForVideo(video, timestamp);
      frames.push(assignFrame(p.landmarks, h.landmarks));
      progressBar.style.width = `${Math.round((i + 1) / model.frames * 100)}%`;
      if (i % 4 === 0) { say(`Reading video frame ${i + 1} of ${model.frames}…`); await new Promise((resolve) => setTimeout(resolve, 0)); }
    }
    const usable = frames.filter(({ valid }) => valid[11] && valid[12] && valid.slice(33).some(Boolean)).length;
    if (usable < model.frames / 2) throw new Error('Landmark tracking was weak. Keep your shoulders and hands fully visible, use brighter light, and try again.');
    const predictions = classify(frames, model);
    showResults(predictions);
    say('Analysis complete. This is an experimental suggestion, not a verified translation.');
  } catch (error) { say(error.message || 'Could not analyze this clip. Try another short video.', true); }
  finally {
    pose?.close(); hands?.close();
    progressWrap.hidden = true;
    try { video.currentTime = 0; } catch { /* An unreadable video already has an error message. */ }
    busy = false; button.disabled = !Number.isFinite(video.duration) || video.duration < 0.5 || video.duration > 30;
  }
}

button.addEventListener('click', analyze);
