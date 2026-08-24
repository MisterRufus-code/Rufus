import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate, spring, Sequence} from 'remotion';
import {Mascot, Footprint, Pose, restPose} from './Mascot';

// ── walk cycle ──────────────────────────────────────────────────────────────
export const CYCLE = 0.8;      // seconds for a full two-step cycle
const THIGH = 50;
const SHIN = 28;
const LEG = THIGH + SHIN;

/** The stride is chosen first and the ground speed is DERIVED from it, never
 *  the other way round. A speed that disagrees with the stride makes the feet
 *  skate, and skating is the first thing that reads as cheap even when every
 *  frame is otherwise clean. */
export const STEP = 46;
const STANCE = 0.62;                       // fraction of the cycle a foot is down
const TRAVEL = 2 * STEP * STANCE;          // how far a planted foot travels under the body
export const WALK_SPEED = (2 * STEP) / CYCLE;
const HIP_H = 65;                          // ankle depth below the hip when standing
const LIFT = 28;                           // how high the swinging foot clears the ground

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** Where the ankle is, relative to the hip, at this point in the leg's own
 *  cycle. Driving the walk from the FOOT and solving the joints backwards is
 *  what makes the planted foot stay planted: during stance the foot moves back
 *  under the body at exactly the speed the body moves forward, so it is nailed
 *  to the ground by construction rather than by a lucky choice of numbers. */
const anklePath = (u: number, bob: number): [number, number] => {
  if (u < STANCE) {
    const k = u / STANCE;
    return [TRAVEL / 2 - TRAVEL * k, HIP_H + bob];
  }
  const k = (u - STANCE) / (1 - STANCE);
  const e = 0.5 - 0.5 * Math.cos(Math.PI * k);
  return [-TRAVEL / 2 + TRAVEL * e, HIP_H + bob - LIFT * Math.sin(Math.PI * k)];
};

/** Two-bone inverse kinematics. Angles come back in this rig's convention —
 *  degrees from straight down — and the joint is always resolved to the same
 *  side, because a knee that bends both ways is the clearest sign nobody looked
 *  at a real walk. Reaching a hand somewhere is the same problem as planting a
 *  foot, so arms use it too: a pose is then a PLACE the hand goes, which is how
 *  a pose is actually described, instead of two angles guessed at until they
 *  look right. */
export const twoBoneIK = (tx: number, ty: number, l1: number, l2: number): [number, number] => {
  const maxReach = l1 + l2 - 1.5;
  let d = Math.hypot(tx, ty);
  let x = tx;
  let y = ty;
  if (d > maxReach) {
    x = (tx / d) * maxReach;
    y = (ty / d) * maxReach;
    d = maxReach;
  }
  const dirDeg = (Math.atan2(-x, y) * 180) / Math.PI;
  const a = Math.acos(Math.min(1, Math.max(-1, (d * d + l1 * l1 - l2 * l2) / (2 * d * l1))));
  const k = Math.acos(Math.min(1, Math.max(-1, (l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2))));
  return [dirDeg - (a * 180) / Math.PI, 180 - (k * 180) / Math.PI];
};

const legIK = (tx: number, ty: number): [number, number] => {
  const maxReach = LEG - 1.5;
  let d = Math.hypot(tx, ty);
  let x = tx;
  let y = ty;
  if (d > maxReach) {
    x = (tx / d) * maxReach;
    y = (ty / d) * maxReach;
    d = maxReach;
  }
  const dirDeg = (Math.atan2(-x, y) * 180) / Math.PI;
  const a = Math.acos(Math.min(1, Math.max(-1, (d * d + THIGH * THIGH - SHIN * SHIN) / (2 * d * THIGH))));
  const k = Math.acos(Math.min(1, Math.max(-1, (THIGH * THIGH + SHIN * SHIN - d * d) / (2 * THIGH * SHIN))));
  const thigh = dirDeg - (a * 180) / Math.PI;
  const shin = 180 - (k * 180) / Math.PI;
  return [thigh, shin];
};

/** Deterministic blinking. A fixed rhythm reads as a machine, so two intervals
 *  that never divide into each other are layered to keep the blinks off any
 *  beat the viewer can predict. */
const blinkAt = (t: number) => {
  const shape = (phase: number) => {
    if (phase < 0 || phase > 0.14) return 0;
    return Math.sin((phase / 0.14) * Math.PI);
  };
  return clamp01(shape(t % 3.7) + shape((t + 1.9) % 5.3));
};

const frac = (v: number) => v - Math.floor(v);

export const walkPose = (t: number): Pose => {
  const ph = (2 * Math.PI * t) / CYCLE;
  const p = restPose();

  // The hip rides highest over a single planted foot and lowest when both are
  // down, which is twice a cycle — the body's rise and fall is a consequence of
  // the legs, not a sine wave laid on top of them.
  const bob = 4 * (0.5 - 0.5 * Math.cos(2 * ph));
  const uL = frac(t / CYCLE);
  const uR = frac(t / CYCLE + 0.5);
  const [lx, ly] = anklePath(uL, bob);
  const [rx, ry] = anklePath(uR, bob);
  p.legL = legIK(lx, ly);
  p.legR = legIK(rx, ry);

  p.y = -bob;
  p.rot = -1.5;                              // a walk leans into where it is going
  p.squash = 1 + 0.03 * Math.sin(2 * ph + 1.0);

  // Paper ripple on its own clock, slower than the steps, so the two never lock
  // into one beat and start reading as a mechanism.
  p.wave = (2 * Math.PI * t) / 1.15;
  p.waveAmp = 7;

  // The free arm swings against its own leg; the glass arm stays up, because it
  // is the half of the silhouette that says what the character does.
  p.armR = [-63 + 20 * Math.sin(ph), 54 + 10 * Math.max(0, Math.sin(ph))];
  p.armL = [120 + 5 * Math.sin(ph + Math.PI), 51 + 4 * Math.sin(ph)];

  // Overlap and follow-through: the hat and the glass arrive a beat after the
  // body does. This is most of what separates an expensive-looking character
  // from a drawing being slid across the screen.
  const lag = ph - 0.9;
  p.hatRot = -7 + 4.5 * Math.sin(lag);
  p.hatLift = 1.8 * Math.sin(2 * lag);
  p.glassRot = -10 + 7 * Math.sin(lag);

  p.blink = blinkAt(t);
  p.look = [1.5 * Math.sin(ph * 0.5), 0];
  return p;
};

export const idlePose = (t: number): Pose => {
  const p = restPose();
  const b = Math.sin((2 * Math.PI * t) / 2.6);       // breathing
  p.y = -2 * b;
  p.squash = 1 + 0.018 * b;
  p.wave = (2 * Math.PI * t) / 2.1;
  p.waveAmp = 3.4;
  p.rot = 0.8 * Math.sin((2 * Math.PI * t) / 3.4);
  p.hatRot = -7 + 1.6 * Math.sin((2 * Math.PI * (t - 0.25)) / 2.6);
  p.hatLift = 0.9 * b;
  p.glassRot = -10 + 4 * Math.sin((2 * Math.PI * (t - 0.3)) / 2.9);
  p.armL = [120 + 2.5 * b, 51 + 2 * b];
  p.armR = [-63 + 2 * b, 54];
  p.blink = blinkAt(t);
  // The eyes drift as though reading something, which is the cheapest possible
  // signal that there is somebody in there.
  p.look = [2.6 * Math.sin((2 * Math.PI * t) / 4.3), 1.2 * Math.sin((2 * Math.PI * t) / 6.1)];
  return p;
};

const SHOULDER: [number, number] = [-146, -34];
const ARM_UPPER = 46;
const ARM_FORE = 42;
const HAND_REST: [number, number] = [-192, -98];   // where the idle pose holds it
const HAND_INSPECT: [number, number] = [-84, -6];  // glass over ONE eye; the other stays visible

/** The character brings the glass up to its own face: the pose for the moment a
 *  script asks the viewer to look closely at something. The hand is aimed at a
 *  PLACE and the joints are solved backwards, so the glass lands where it is
 *  supposed to instead of somewhere two guessed angles happen to put it. */
export const inspectPose = (t: number, progress: number): Pose => {
  const p = idlePose(t);
  const e = progress * progress * (3 - 2 * progress);   // smoothstep
  const hx = HAND_REST[0] + (HAND_INSPECT[0] - HAND_REST[0]) * e;
  const hy = HAND_REST[1] + (HAND_INSPECT[1] - HAND_REST[1]) * e;
  p.armL = twoBoneIK(hx - SHOULDER[0], hy - SHOULDER[1], ARM_UPPER, ARM_FORE);
  // -90 in this convention points the glass straight along +x, at the face.
  const aimed = -90 - (p.armL[0] + p.armL[1]);
  p.glassRot = -10 + (aimed + 10) * e;
  p.armFront = e > 0.3;
  p.magnify = e > 0.55;
  p.rot = p.rot - 3.5 * e;
  p.brow = 'raised';
  p.mouth = e > 0.6 ? 'flat' : 'smile';
  p.look = [7 * e, -2 * e];
  return p;
};

// ── the trail ───────────────────────────────────────────────────────────────
/** Footprints are planted where a foot actually touched down, not sprinkled on
 *  a timer, so the trail lines up with the walk instead of merely accompanying
 *  it. They persist: the trail is the channel's through-line, and it is the
 *  same shape as the symbol in the note's four corners. */
type Print = {x: number; y: number; rot: number; born: number};

export const printsUpTo = (t: number, originX: number, speed: number, scale: number, groundY: number): Print[] => {
  const out: Print[] = [];
  const HIP_X = 49;
  const SHOE_MID = 20;   // the sole reaches outward from the ankle it hangs on
  // A print is planted at heel strike, at the exact world position the foot
  // lands on — which is why the trail lines up with the walk instead of merely
  // accompanying it.
  for (let k = 0; k < 200; k++) {
    const side = k % 2 === 0 ? -1 : 1;
    const tc = (k % 2 === 0 ? 0 : 0.5) * CYCLE + Math.floor(k / 2) * CYCLE;
    if (tc > t) break;
    out.push({
      x: originX + speed * tc + (side * HIP_X + TRAVEL / 2 + side * SHOE_MID) * scale,
      y: groundY,
      rot: side * 7 + (k % 3) * 2,
      born: tc,
    });
  }
  return out;
};

const Trail: React.FC<{prints: Print[]; t: number; scale: number}> = ({prints, t, scale}) => (
  <>
    {prints.map((pr, i) => {
      const age = t - pr.born;
      const pop = clamp01(age / 0.18);
      return (
        <Footprint
          key={i}
          x={pr.x}
          y={pr.y}
          rot={pr.rot}
          scale={scale * 2.1 * (0.78 + 0.22 * pop)}
          opacity={interpolate(age, [0, 0.18, 5], [0, 1, 0.5], {extrapolateRight: 'clamp'})}
        />
      );
    })}
  </>
);

// ── scenes ──────────────────────────────────────────────────────────────────
const Paper: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill style={{backgroundColor: '#F3F6F1'}}>{children}</AbsoluteFill>
);

export const MascotWalk: React.FC<{scale?: number; showTrail?: boolean}> = ({
  scale = 1.15, showTrail = true,
}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const speed = WALK_SPEED * scale;
  const startX = -220 * scale;
  const groundY = height * 0.72;
  const originY = groundY - 173 * scale;
  const x = startX + speed * t;
  const pose = walkPose(t);

  return (
    <Paper>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {showTrail ? <Trail prints={printsUpTo(t, startX, speed, scale, groundY)} t={t} scale={scale} /> : null}
        <g transform={`translate(${x},${originY + pose.y * scale})`}>
          <Mascot pose={{...pose, y: 0}} scale={scale} />
        </g>
      </svg>
    </Paper>
  );
};

export const MascotIdle: React.FC<{scale?: number}> = ({scale = 1.3}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const pose = idlePose(t);
  return (
    <Paper>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <g transform={`translate(${width / 2},${height / 2})`}>
          <Mascot pose={pose} scale={scale} />
        </g>
      </svg>
    </Paper>
  );
};

export const MascotInspect: React.FC<{scale?: number}> = ({scale = 1.3}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const lift = spring({frame: frame - Math.round(0.4 * fps), fps, config: {damping: 14, mass: 0.8}});
  const pose = inspectPose(t, lift);
  return (
    <Paper>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <g transform={`translate(${width / 2},${height / 2})`}>
          <Mascot pose={pose} scale={scale} />
        </g>
      </svg>
    </Paper>
  );
};

/** Paper falls the way paper falls: it does not drop, it swings on the air and
 *  then settles. The landing squash is what sells the weight of a thing that
 *  has almost none. */
export const MascotEntrance: React.FC<{scale?: number}> = ({scale = 1.3}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const fall = clamp01(t / 1.25);
  const restY = height / 2;
  const y = interpolate(fall, [0, 1], [-height * 0.42, restY]);
  const flutter = (1 - fall) * 70 * Math.sin(t * 6.0);
  const settle = spring({frame: frame - Math.round(1.25 * fps), fps, config: {damping: 9, mass: 0.5}});

  const pose = idlePose(Math.max(0, t - 1.25));
  pose.rot = (1 - fall) * 16 * Math.sin(t * 5.2) + pose.rot * fall;
  pose.wave = t * 9;
  pose.waveAmp = 5 + (1 - fall) * 9;
  pose.squash = fall < 1 ? 1 : interpolate(settle, [0, 0.35, 1], [0.82, 1.06, 1], {extrapolateRight: 'clamp'});
  pose.armL = [120 + (1 - fall) * 26, 51];
  pose.armR = [-63 - (1 - fall) * 26, 54];
  pose.legL = [9 + (1 - fall) * 18, -6];
  pose.legR = [-9 - (1 - fall) * 18, 6];
  pose.mouth = fall < 1 ? 'open' : 'grin';
  pose.brow = fall < 1 ? 'raised' : 'delighted';

  return (
    <Paper>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <g transform={`translate(${width / 2 + flutter},${y})`}>
          <Mascot pose={pose} scale={scale} />
        </g>
      </svg>
    </Paper>
  );
};

export const MascotShowreel: React.FC = () => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill>
      <Sequence durationInFrames={Math.round(3.2 * fps)}><MascotEntrance /></Sequence>
      <Sequence from={Math.round(3.2 * fps)} durationInFrames={Math.round(3.0 * fps)}><MascotIdle /></Sequence>
      <Sequence from={Math.round(6.2 * fps)} durationInFrames={Math.round(3.0 * fps)}><MascotInspect /></Sequence>
      <Sequence from={Math.round(9.2 * fps)} durationInFrames={Math.round(6.0 * fps)}><MascotWalk /></Sequence>
    </AbsoluteFill>
  );
};
