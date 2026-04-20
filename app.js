import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34";

const TASKS_VISION_VERSION = "0.10.34";
const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

const HAND_COLORS = [
  {
    line: [132, 244, 255],
    point: [235, 252, 255]
  },
  {
    line: [154, 255, 198],
    point: [243, 255, 247]
  }
];

const config = Object.freeze({
  gravity: 1280,
  floorOffset: 36,
  wallBounce: 0.62,
  floorBounce: 0.16,
  airDrag: 0.12,
  ballRadius: 44,
  landmarkSmoothing: 0.46,
  handSmoothing: 0.24,
  velocitySmoothing: 0.24,
  pinchCloseRatio: 0.42,
  pinchOpenRatio: 0.58,
  pickupRadius: 88,
  throwScale: 0.96,
  maxGrabRelativeSpeed: 980,
  handTimeoutMs: 160,
  supportCaptureDepth: 0.5,
  supportCarry: 0.24,
  supportOpenThreshold: 1.22,
  collisionPasses: 2
});

const video = document.getElementById("webcam");
const messageEl = document.getElementById("message");

function createHandState(id) {
  return {
    id,
    detected: false,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    palmX: 0,
    palmY: 0,
    pinchX: 0,
    pinchY: 0,
    handScale: 1,
    pinchRatio: Infinity,
    pinchClosed: false,
    openAmount: 0,
    supportA: { x: 0, y: 0 },
    supportB: { x: 0, y: 0 },
    supportNormalX: 0,
    supportNormalY: -1,
    supportTangentX: 1,
    supportTangentY: 0,
    colliders: [],
    landmarks: [],
    lastSeenAt: 0
  };
}

const state = {
  landmarker: null,
  cameraReady: false,
  initializationState: "idle",
  lastVideoTime: -1,
  floorY: 0,
  retryListenerArmed: false,
  hands: [createHandState(0), createHandState(1)],
  ball: {
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    radius: config.ballRadius,
    heldBy: null,
    supportedBy: null,
    renderStretchX: 1,
    renderStretchY: 1,
    glowPhase: Math.random() * Math.PI * 2
  }
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function lerp(start, end, amount) {
  return start + (end - start) * amount;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function dot(ax, ay, bx, by) {
  return ax * bx + ay * by;
}

function normalizeVector(x, y, fallbackX, fallbackY) {
  const length = Math.hypot(x, y);
  if (length < 1e-5) {
    return { x: fallbackX, y: fallbackY };
  }

  return {
    x: x / length,
    y: y / length
  };
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5
  };
}

function averagePoints(points) {
  const sum = points.reduce(
    (acc, point) => {
      acc.x += point.x;
      acc.y += point.y;
      return acc;
    },
    { x: 0, y: 0 }
  );

  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
}

function copyPoint(point) {
  return {
    x: point.x,
    y: point.y,
    z: point.z ?? 0
  };
}

function closestPointOnSegment(point, a, b, extension = 0) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const lengthSq = abx * abx + aby * aby;

  if (lengthSq < 1e-5) {
    return { x: a.x, y: a.y, t: 0 };
  }

  const apx = point.x - a.x;
  const apy = point.y - a.y;
  const rawT = dot(apx, apy, abx, aby) / lengthSq;
  const t = clamp(rawT, -extension, 1 + extension);

  return {
    x: a.x + abx * t,
    y: a.y + aby * t,
    t
  };
}

function setMessage(message) {
  messageEl.textContent = message;
}

function resetBall(width, height) {
  state.ball.x = width * 0.5;
  state.ball.y = Math.min(height * 0.34, height - config.floorOffset - state.ball.radius - 80);
  state.ball.vx = 0;
  state.ball.vy = 0;
  state.ball.heldBy = null;
  state.ball.supportedBy = null;
  state.ball.renderStretchX = 1;
  state.ball.renderStretchY = 1;
}

function waitForVideoReady() {
  return new Promise((resolve) => {
    if (video.readyState >= 2) {
      resolve();
      return;
    }

    video.addEventListener("loadeddata", resolve, { once: true });
  });
}

async function ensureLandmarker() {
  if (state.landmarker) {
    return state.landmarker;
  }

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  const sharedOptions = {
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.55
  };

  try {
    state.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET_PATH,
        delegate: "GPU"
      },
      ...sharedOptions
    });
  } catch (gpuError) {
    console.warn("GPU delegate unavailable, falling back to CPU.", gpuError);
    state.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET_PATH
      },
      ...sharedOptions
    });
  }

  return state.landmarker;
}

function armRetryListener() {
  if (state.retryListenerArmed) {
    return;
  }

  state.retryListenerArmed = true;
  window.addEventListener(
    "pointerdown",
    () => {
      state.retryListenerArmed = false;
      startCamera();
    },
    { once: true }
  );
}

async function startCamera() {
  if (state.initializationState === "starting" || state.cameraReady) {
    return;
  }

  state.initializationState = "starting";
  setMessage("Allow camera access if your browser asks.");

  try {
    await ensureLandmarker();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });

    video.srcObject = stream;
    await video.play();
    await waitForVideoReady();

    state.cameraReady = true;
    state.initializationState = "live";
    setMessage("");
  } catch (error) {
    console.error(error);
    state.cameraReady = false;
    state.initializationState = "error";

    if (error?.name === "NotAllowedError") {
      setMessage("Camera access blocked. Allow access, then tap anywhere to retry.");
      armRetryListener();
      return;
    }

    setMessage("Camera unavailable. Check permissions, then tap anywhere to retry.");
    armRetryListener();
  }
}

function mirrorLandmarks(landmarks, width, height) {
  return landmarks.map((landmark) => ({
    x: (1 - landmark.x) * width,
    y: landmark.y * height,
    z: landmark.z
  }));
}

function trackingCost(hand, rawLandmarks) {
  if (!hand.detected) {
    return 0;
  }

  const palm = averagePoints([rawLandmarks[0], rawLandmarks[5], rawLandmarks[9], rawLandmarks[13], rawLandmarks[17]]);
  return distance({ x: hand.palmX, y: hand.palmY }, palm);
}

function buildHandColliders(hand, landmarks, palm) {
  const scale = hand.handScale;
  const colliders = [
    { x: palm.x, y: palm.y, radius: scale * 0.34, bounce: 0.02, carry: 0.24 },
    { x: landmarks[0].x, y: landmarks[0].y, radius: scale * 0.18, bounce: 0.1, carry: 0.16 }
  ];

  for (const index of [5, 9, 13, 17]) {
    colliders.push({
      x: landmarks[index].x,
      y: landmarks[index].y,
      radius: scale * 0.15,
      bounce: 0.06,
      carry: 0.18
    });
  }

  for (const index of [6, 10, 14, 18]) {
    colliders.push({
      x: landmarks[index].x,
      y: landmarks[index].y,
      radius: scale * 0.11,
      bounce: 0.1,
      carry: 0.16
    });
  }

  for (const index of [4, 8, 12, 16, 20]) {
    colliders.push({
      x: landmarks[index].x,
      y: landmarks[index].y,
      radius: scale * 0.1,
      bounce: 0.14,
      carry: 0.16
    });
  }

  return colliders;
}

function refreshHandGeometry(hand, dt, isNewHand) {
  const landmarks = hand.landmarks;
  const palm = averagePoints([landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]);
  const pinch = midpoint(landmarks[4], landmarks[8]);
  const fingertips = averagePoints([landmarks[8], landmarks[12], landmarks[16], landmarks[20]]);
  const handScale = (distance(landmarks[0], landmarks[9]) + distance(landmarks[5], landmarks[17])) * 0.5 || 1;
  const pinchRatio = distance(landmarks[4], landmarks[8]) / handScale;
  const openAmount =
    ([8, 12, 16, 20].reduce((sum, index) => sum + distance(landmarks[index], landmarks[0]), 0) / 4) / handScale;

  if (hand.pinchClosed) {
    if (pinchRatio > config.pinchOpenRatio) {
      hand.pinchClosed = false;
    }
  } else if (pinchRatio < config.pinchCloseRatio) {
    hand.pinchClosed = true;
  }

  const target = hand.pinchClosed ? pinch : palm;
  const previousX = isNewHand ? target.x : hand.x;
  const previousY = isNewHand ? target.y : hand.y;
  const smoothing = hand.pinchClosed ? 0.4 : config.handSmoothing;

  hand.x = isNewHand ? target.x : lerp(hand.x, target.x, smoothing);
  hand.y = isNewHand ? target.y : lerp(hand.y, target.y, smoothing);

  const safeDt = Math.max(dt, 1 / 240);
  const instantVx = (hand.x - previousX) / safeDt;
  const instantVy = (hand.y - previousY) / safeDt;

  hand.vx = isNewHand ? 0 : lerp(hand.vx, instantVx, config.velocitySmoothing);
  hand.vy = isNewHand ? 0 : lerp(hand.vy, instantVy, config.velocitySmoothing);
  hand.palmX = palm.x;
  hand.palmY = palm.y;
  hand.pinchX = pinch.x;
  hand.pinchY = pinch.y;
  hand.handScale = handScale;
  hand.pinchRatio = pinchRatio;
  hand.openAmount = openAmount;

  const tangent = normalizeVector(landmarks[17].x - landmarks[5].x, landmarks[17].y - landmarks[5].y, 1, 0);
  const supportNormal = normalizeVector(
    fingertips.x - landmarks[0].x,
    fingertips.y - landmarks[0].y,
    0,
    -1
  );
  const supportCenter = averagePoints([landmarks[5], landmarks[9], landmarks[13], landmarks[17]]);
  const supportHalfWidth = distance(landmarks[5], landmarks[17]) * 0.62;
  const supportOffset = handScale * 0.12;

  hand.supportTangentX = tangent.x;
  hand.supportTangentY = tangent.y;
  hand.supportNormalX = supportNormal.x;
  hand.supportNormalY = supportNormal.y;
  hand.supportA = {
    x: supportCenter.x + supportNormal.x * supportOffset - tangent.x * supportHalfWidth,
    y: supportCenter.y + supportNormal.y * supportOffset - tangent.y * supportHalfWidth
  };
  hand.supportB = {
    x: supportCenter.x + supportNormal.x * supportOffset + tangent.x * supportHalfWidth,
    y: supportCenter.y + supportNormal.y * supportOffset + tangent.y * supportHalfWidth
  };
  hand.colliders = buildHandColliders(hand, landmarks, palm);
}

function applyRawHandToSlot(hand, rawLandmarks, dt, nowMs) {
  const isNewHand = !hand.detected || hand.landmarks.length !== rawLandmarks.length;

  if (isNewHand) {
    hand.landmarks = rawLandmarks.map(copyPoint);
  } else {
    hand.landmarks = rawLandmarks.map((point, index) => ({
      x: lerp(hand.landmarks[index].x, point.x, config.landmarkSmoothing),
      y: lerp(hand.landmarks[index].y, point.y, config.landmarkSmoothing),
      z: lerp(hand.landmarks[index].z ?? 0, point.z ?? 0, config.landmarkSmoothing)
    }));
  }

  hand.detected = true;
  refreshHandGeometry(hand, dt, isNewHand);
  hand.lastSeenAt = nowMs;
}

function clearHand(hand) {
  hand.detected = false;
  hand.landmarks = [];
  hand.colliders = [];
  hand.pinchClosed = false;
  hand.pinchRatio = Infinity;
  hand.openAmount = 0;
}

function applyDetectedHands(rawHands, dt, nowMs) {
  const hands = state.hands;
  const assigned = new Array(hands.length).fill(false);

  if (rawHands.length === 1) {
    const slotIndex = trackingCost(hands[0], rawHands[0]) <= trackingCost(hands[1], rawHands[0]) ? 0 : 1;
    applyRawHandToSlot(hands[slotIndex], rawHands[0], dt, nowMs);
    assigned[slotIndex] = true;
  } else if (rawHands.length >= 2) {
    const limited = rawHands.slice(0, 2);
    const directCost = trackingCost(hands[0], limited[0]) + trackingCost(hands[1], limited[1]);
    const swappedCost = trackingCost(hands[0], limited[1]) + trackingCost(hands[1], limited[0]);
    const ordered = directCost <= swappedCost ? limited : [limited[1], limited[0]];

    applyRawHandToSlot(hands[0], ordered[0], dt, nowMs);
    applyRawHandToSlot(hands[1], ordered[1], dt, nowMs);
    assigned[0] = true;
    assigned[1] = true;
  }

  for (let index = 0; index < hands.length; index += 1) {
    const hand = hands[index];
    if (assigned[index]) {
      continue;
    }

    if (nowMs - hand.lastSeenAt > config.handTimeoutMs) {
      clearHand(hand);
    }
  }
}

function updateHandTracking(nowMs, dt, width, height) {
  if (!state.landmarker || !state.cameraReady || video.readyState < 2) {
    for (const hand of state.hands) {
      clearHand(hand);
    }
    return;
  }

  if (video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const results = state.landmarker.detectForVideo(video, nowMs);
    const landmarksList = results.landmarks ?? [];
    const rawHands = landmarksList.map((landmarks) => mirrorLandmarks(landmarks, width, height));
    applyDetectedHands(rawHands, dt, nowMs);
    return;
  }

  for (const hand of state.hands) {
    if (nowMs - hand.lastSeenAt > config.handTimeoutMs) {
      clearHand(hand);
    }
  }
}

function releaseHeldBall(hand) {
  state.ball.heldBy = null;
  state.ball.vx = hand.vx * config.throwScale;
  state.ball.vy = hand.vy * config.throwScale;
}

function attachBallToHand(hand, width) {
  state.ball.heldBy = hand.id;
  state.ball.supportedBy = null;
  state.ball.x = clamp(hand.pinchX, state.ball.radius + 12, width - state.ball.radius - 12);
  state.ball.y = clamp(
    hand.pinchY - state.ball.radius * 0.08,
    state.ball.radius + 12,
    state.floorY - state.ball.radius
  );
  state.ball.vx = hand.vx;
  state.ball.vy = hand.vy;
}

function resolveWorldCollisions(ball, width) {
  if (ball.x - ball.radius < 0) {
    ball.x = ball.radius;
    ball.vx = Math.abs(ball.vx) * config.wallBounce;
  } else if (ball.x + ball.radius > width) {
    ball.x = width - ball.radius;
    ball.vx = -Math.abs(ball.vx) * config.wallBounce;
  }

  if (ball.y + ball.radius > state.floorY) {
    const impactSpeed = Math.abs(ball.vy);
    ball.y = state.floorY - ball.radius;
    ball.vy = impactSpeed > 160 ? -impactSpeed * config.floorBounce : 0;
    ball.vx *= impactSpeed > 160 ? 0.92 : 0.8;

    if (Math.abs(ball.vx) < 5) {
      ball.vx = 0;
    }
  }

  if (ball.y < ball.radius) {
    ball.y = ball.radius;
    ball.vy = Math.abs(ball.vy) * 0.3;
  }
}

function constrainBallToWorld(ball, width) {
  ball.x = clamp(ball.x, ball.radius, width - ball.radius);
  ball.y = clamp(ball.y, ball.radius, state.floorY - ball.radius);
}

function getSupportCandidate(ball, hand) {
  if (!hand.detected || hand.openAmount < config.supportOpenThreshold) {
    return null;
  }

  const closest = closestPointOnSegment({ x: ball.x, y: ball.y }, hand.supportA, hand.supportB, 0.18);
  const offsetX = ball.x - closest.x;
  const offsetY = ball.y - closest.y;
  const signedDistance = dot(offsetX, offsetY, hand.supportNormalX, hand.supportNormalY);
  const relativeNormalSpeed = dot(
    ball.vx - hand.vx,
    ball.vy - hand.vy,
    hand.supportNormalX,
    hand.supportNormalY
  );
  const captureDepth = ball.radius + hand.handScale * config.supportCaptureDepth;

  if (signedDistance < -hand.handScale * 0.16 || signedDistance > captureDepth) {
    return null;
  }

  if (relativeNormalSpeed > 220) {
    return null;
  }

  return {
    hand,
    closest,
    signedDistance,
    score:
      Math.abs(signedDistance - ball.radius) +
      Math.abs(relativeNormalSpeed) * 0.04 -
      Math.max(0, hand.openAmount - config.supportOpenThreshold) * 18
  };
}

function applySupportCandidate(ball, candidate, width) {
  const { hand, closest } = candidate;
  const nx = hand.supportNormalX;
  const ny = hand.supportNormalY;
  const tx = hand.supportTangentX;
  const ty = hand.supportTangentY;
  const targetX = closest.x + nx * ball.radius;
  const targetY = closest.y + ny * ball.radius;
  const handTangentSpeed = dot(hand.vx, hand.vy, tx, ty);
  const ballTangentSpeed = dot(ball.vx, ball.vy, tx, ty);
  const handNormalSpeed = Math.max(dot(hand.vx, hand.vy, nx, ny), 0);
  const nextTangentSpeed = lerp(ballTangentSpeed, handTangentSpeed, config.supportCarry);

  ball.x = clamp(targetX, ball.radius, width - ball.radius);
  ball.y = clamp(targetY, ball.radius, state.floorY - ball.radius);
  ball.vx = tx * nextTangentSpeed + nx * handNormalSpeed;
  ball.vy = ty * nextTangentSpeed + ny * handNormalSpeed;
  ball.supportedBy = hand.id;
}

function resolveBallAgainstHandColliders(ball, hand) {
  if (!hand.detected) {
    return;
  }

  for (const collider of hand.colliders) {
    const dx = ball.x - collider.x;
    const dy = ball.y - collider.y;
    const minDistance = ball.radius + collider.radius;
    const distanceSq = dx * dx + dy * dy;

    if (distanceSq >= minDistance * minDistance) {
      continue;
    }

    const normal = normalizeVector(dx, dy, 0, -1);
    ball.x = collider.x + normal.x * minDistance;
    ball.y = collider.y + normal.y * minDistance;

    const relativeNormalSpeed = dot(ball.vx - hand.vx, ball.vy - hand.vy, normal.x, normal.y);
    if (relativeNormalSpeed < 0) {
      ball.vx -= (1 + collider.bounce) * relativeNormalSpeed * normal.x;
      ball.vy -= (1 + collider.bounce) * relativeNormalSpeed * normal.y;
      ball.vx = lerp(ball.vx, hand.vx, collider.carry);
      ball.vy = lerp(ball.vy, hand.vy, collider.carry);
    }
  }
}

function findBestSupportCandidate(ball) {
  let best = null;

  for (const hand of state.hands) {
    const candidate = getSupportCandidate(ball, hand);
    if (!candidate) {
      continue;
    }

    if (!best || candidate.score < best.score) {
      best = candidate;
    }
  }

  return best;
}

function findGrabCandidate(ball) {
  let best = null;

  for (const hand of state.hands) {
    if (!hand.detected || !hand.pinchClosed) {
      continue;
    }

    const grabDistance = distance({ x: ball.x, y: ball.y }, { x: hand.pinchX, y: hand.pinchY });
    const relativeSpeed = Math.hypot(ball.vx - hand.vx, ball.vy - hand.vy);
    const reach = config.pickupRadius + hand.handScale * 0.12;

    if (grabDistance > reach) {
      continue;
    }

    if (relativeSpeed > config.maxGrabRelativeSpeed && grabDistance > hand.handScale * 0.38) {
      continue;
    }

    const score = grabDistance + relativeSpeed * 0.03;
    if (!best || score < best.score) {
      best = { hand, score };
    }
  }

  return best;
}

function updateBallStretch(ball, dt) {
  const grounded = ball.y + ball.radius >= state.floorY - 0.5 && Math.abs(ball.vy) < 1;
  if (grounded) {
    ball.vx *= Math.pow(0.88, dt * 60);
    if (Math.abs(ball.vx) < 2) {
      ball.vx = 0;
    }
  }

  const speed = Math.hypot(ball.vx, ball.vy);
  let targetStretchX = 1 + Math.min(speed / 1800, 0.14);
  let targetStretchY = 1 - Math.min(speed / 2600, 0.1);

  if (grounded) {
    targetStretchX = 1.08;
    targetStretchY = 0.93;
  }

  ball.renderStretchX = lerp(ball.renderStretchX, targetStretchX, 0.16);
  ball.renderStretchY = lerp(ball.renderStretchY, targetStretchY, 0.16);
}

function updateBall(dt, nowMs, width) {
  const ball = state.ball;
  ball.glowPhase = nowMs * 0.0025;
  ball.supportedBy = null;

  if (ball.heldBy !== null) {
    const hand = state.hands[ball.heldBy];
    if (!hand.detected || !hand.pinchClosed) {
      releaseHeldBall(hand);
    } else {
      attachBallToHand(hand, width);
      ball.renderStretchX = lerp(ball.renderStretchX, 1.02, 0.18);
      ball.renderStretchY = lerp(ball.renderStretchY, 0.98, 0.18);
      return;
    }
  }

  ball.vy += config.gravity * dt;
  ball.vx *= 1 - config.airDrag * dt;
  ball.x += ball.vx * dt;
  ball.y += ball.vy * dt;

  resolveWorldCollisions(ball, width);

  const supportCandidate = findBestSupportCandidate(ball);
  if (supportCandidate) {
    applySupportCandidate(ball, supportCandidate, width);
  }

  for (let pass = 0; pass < config.collisionPasses; pass += 1) {
    for (const hand of state.hands) {
      if (ball.supportedBy === hand.id) {
        continue;
      }
      resolveBallAgainstHandColliders(ball, hand);
    }
  }

  if (ball.supportedBy === null) {
    const lateSupportCandidate = findBestSupportCandidate(ball);
    if (lateSupportCandidate) {
      applySupportCandidate(ball, lateSupportCandidate, width);
    }
  }

  constrainBallToWorld(ball, width);

  const grabCandidate = findGrabCandidate(ball);
  if (grabCandidate) {
    attachBallToHand(grabCandidate.hand, width);
    return;
  }

  updateBallStretch(ball, dt);
}

function drawHand(p, hand) {
  if (!hand.detected || hand.landmarks.length === 0) {
    return;
  }

  const color = HAND_COLORS[hand.id % HAND_COLORS.length];

  p.stroke(0, 0, 0, 90);
  p.strokeWeight(6);
  for (const [start, end] of HAND_CONNECTIONS) {
    const from = hand.landmarks[start];
    const to = hand.landmarks[end];
    p.line(from.x, from.y, to.x, to.y);
  }

  p.stroke(...color.line, hand.pinchClosed ? 240 : 210);
  p.strokeWeight(hand.pinchClosed ? 3.8 : 3);
  for (const [start, end] of HAND_CONNECTIONS) {
    const from = hand.landmarks[start];
    const to = hand.landmarks[end];
    p.line(from.x, from.y, to.x, to.y);
  }

  for (const point of hand.landmarks) {
    p.noStroke();
    p.fill(0, 0, 0, 90);
    p.circle(point.x, point.y, 13);
    p.fill(...color.point, 230);
    p.circle(point.x, point.y, 7);
  }
}

function drawBall(p, nowMs) {
  const ball = state.ball;
  const shadowDistance = clamp((state.floorY - ball.y) / 300, 0.24, 1);
  const shadowWidth = ball.radius * 1.6 * shadowDistance;
  const shadowHeight = ball.radius * 0.42 * shadowDistance;
  const glow = 0.65 + Math.sin(nowMs * 0.004 + ball.glowPhase) * 0.12;

  p.noStroke();
  p.fill(0, 0, 0, 56);
  p.ellipse(ball.x, state.floorY + 7, shadowWidth, shadowHeight);

  p.push();
  p.translate(ball.x, ball.y);
  p.rotate(ball.heldBy !== null ? 0 : Math.atan2(ball.vy, ball.vx + 0.001) * 0.08);
  p.scale(ball.renderStretchX, ball.renderStretchY);

  p.noStroke();
  p.fill(255, 147, 92, 56 * glow);
  p.circle(0, 0, ball.radius * 2.7);
  p.fill(255, 166, 112, 84 * glow);
  p.circle(0, 0, ball.radius * 2.1);
  p.fill(255, 141, 73);
  p.circle(0, 0, ball.radius * 2);
  p.stroke(255, 243, 224, 150);
  p.strokeWeight(1.6);
  p.noFill();
  p.circle(0, 0, ball.radius * 1.95);
  p.noStroke();
  p.fill(255, 190, 118);
  p.circle(-ball.radius * 0.24, -ball.radius * 0.24, ball.radius * 0.7);
  p.fill(255, 233, 204, 190);
  p.circle(-ball.radius * 0.34, -ball.radius * 0.36, ball.radius * 0.25);

  p.pop();
}

function installSketch() {
  new window.p5((p) => {
    p.setup = () => {
      const canvas = p.createCanvas(window.innerWidth, window.innerHeight);
      canvas.parent("canvas-root");
      p.pixelDensity(1);
      p.frameRate(60);
      resetBall(p.width, p.height);
    };

    p.draw = () => {
      const nowMs = performance.now();
      const dt = Math.min(p.deltaTime / 1000, 1 / 24);

      state.floorY = p.height - config.floorOffset;

      if (state.cameraReady) {
        p.clear();
      } else {
        p.background(5, 8, 11);
      }

      updateHandTracking(nowMs, dt, p.width, p.height);
      updateBall(dt, nowMs, p.width);

      for (const hand of state.hands) {
        drawHand(p, hand);
      }

      drawBall(p, nowMs);
    };

    p.windowResized = () => {
      p.resizeCanvas(window.innerWidth, window.innerHeight);
      state.ball.x = clamp(state.ball.x || p.width * 0.5, state.ball.radius, p.width - state.ball.radius);
      state.ball.y = clamp(
        state.ball.y || p.height * 0.34,
        state.ball.radius,
        p.height - config.floorOffset - state.ball.radius
      );
    };
  });
}

function bootstrap() {
  installSketch();

  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("This browser does not support camera access.");
    return;
  }

  startCamera();
}

bootstrap();
