import React from 'react';
/**
 * The Paper Trails mascot, as a rig rather than a picture.
 *
 * The static SVG in assets/brand is a POSE: the limbs are baked into path
 * coordinates, so animating it would mean interpolating path data, which comes
 * out rubbery. Here every limb is a nested group rotating about a real joint,
 * so a walk is a handful of angles over time and the character keeps its exact
 * proportions in every frame. That is also why the mascot is drawn and not
 * generated: a logo has to land on the same pixels every time, and diffusion
 * cannot do that.
 *
 * Coordinate space: the origin is the CENTRE OF THE NOTE, which is the centre
 * of the body, because the note IS the body — there is no separate head. The
 * face lives inside the note's portrait oval.
 */

// ── palette ─────────────────────────────────────────────────────────────────
const OUTLINE = '#141414';
const GREEN = '#2F7D4E';
const LIMB = '#3E9160';
const PALE = '#E8F5EA';
const DEEP = '#1E5C39';
const HAT = '#3B3227';
const HATBAND = '#241D16';
const SHOE = '#33312E';
const BRASS = '#C9A227';
const WOOD = '#6B4A2B';
const LIMB_FAR = '#276641';
const SHOE_FAR = '#232220';
const TRAIL_NEAR = '#6FB68B';

// ── skeleton dimensions (note-local units) ──────────────────────────────────
export const NOTE_W = 296;
export const NOTE_H = 176;
const HALF_W = NOTE_W / 2;
const HALF_H = NOTE_H / 2;

const SHOULDER_Y = -34;
const SHOULDER_X = HALF_W - 2;
const HIP_X = 49;
const HIP_Y = HALF_H;

const UPPER_ARM = 46;
const FOREARM = 42;
const THIGH = 50;
const SHIN = 28;

/** Every angle in this file is degrees from STRAIGHT DOWN, so 0 is a limb
 *  hanging at rest and the numbers read the way a pose reads. */
export type Pose = {
  x: number;
  y: number;
  rot: number;
  squash: number;      // >1 tall and thin, <1 short and wide
  wave: number;        // travelling ripple through the paper, radians
  waveAmp: number;
  armL: [number, number];
  armR: [number, number];
  legL: [number, number];
  legR: [number, number];
  hatRot: number;
  hatLift: number;
  glassRot: number;
  blink: number;       // 0 open, 1 shut
  look: [number, number];
  brow: 'level' | 'raised' | 'curious' | 'worried' | 'delighted';
  /** The glass hand crosses in FRONT of the note when it is raised to the
   *  face — behind it, the hand disappears and the glass reads as floating. */
  armFront?: boolean;
  /** The lens is over the face: show the eye behind it, enlarged. */
  magnify?: boolean;
  mouth: 'smile' | 'flat' | 'open' | 'grin';
};

export const restPose = (): Pose => ({
  x: 0, y: 0, rot: 0, squash: 1, wave: 0, waveAmp: 0,
  armL: [120, 51], armR: [-63, 54],
  legL: [9, -6], legR: [-9, 6],
  hatRot: -7, hatLift: 0, glassRot: -10,
  blink: 0, look: [0, 0], brow: 'curious', mouth: 'smile',
});

// ── bones ───────────────────────────────────────────────────────────────────
type Pass = 'outline' | 'fill';

/** Outlines for the whole skeleton are drawn before any fill. Drawing each
 *  bone outline-then-fill leaves a black notch where the next bone's outline
 *  cap lands on the previous bone's green — the seam is only invisible if the
 *  passes are separated. */
const Bone: React.FC<{len: number; pass: Pass; far?: boolean}> = ({len, pass, far}) => (
  <line
    x1={0} y1={0} x2={0} y2={len}
    stroke={pass === 'outline' ? OUTLINE : far ? LIMB_FAR : LIMB}
    strokeWidth={pass === 'outline' ? 18 : 9}
    strokeLinecap="round"
  />
);

const Hand: React.FC<{pass: Pass; far?: boolean}> = ({pass, far}) =>
  pass === 'outline' ? (
    <circle cx={0} cy={0} r={15} fill={OUTLINE} stroke={OUTLINE} strokeWidth={5} />
  ) : (
    <circle cx={0} cy={0} r={13} fill={far ? LIMB_FAR : LIMB} />
  );

/** Splayed vintage shoes, the only brown at the bottom of the frame. Mirrored
 *  for the far foot so the pair points outward the way a silent-film walk reads
 *  at any size. */
const Shoe: React.FC<{flip: boolean; pass: Pass; far?: boolean}> = ({flip, pass, far}) => (
  <g transform={flip ? 'scale(-1,1)' : undefined}>
    <path
      d="M0,-12 v18 c0,9 -7,13 -17,13 h-24 c-11,0 -16,-5 -16,-12 c0,-8 7,-11 16,-14 l18,-7 z"
      fill={pass === 'outline' ? OUTLINE : far ? SHOE_FAR : SHOE}
      stroke={pass === 'outline' ? OUTLINE : 'none'}
      strokeWidth={pass === 'outline' ? 5 : 0}
      strokeLinejoin="round"
    />
  </g>
);

const Arm: React.FC<{pose: Pose; side: 'L' | 'R'; pass: Pass}> = ({pose, side, pass}) => {
  const [upper, fore] = side === 'L' ? pose.armL : pose.armR;
  const sx = side === 'L' ? -SHOULDER_X : SHOULDER_X;
  const far = side === 'R';
  return (
    <g transform={`translate(${sx},${SHOULDER_Y}) rotate(${upper})`}>
      <Bone len={UPPER_ARM} pass={pass} far={far} />
      <g transform={`translate(0,${UPPER_ARM}) rotate(${fore})`}>
        <Bone len={FOREARM} pass={pass} far={far} />
        <g transform={`translate(0,${FOREARM})`}>
          <Hand pass={pass} far={far} />
        </g>
      </g>
    </g>
  );
};

const Leg: React.FC<{pose: Pose; side: 'L' | 'R'; pass: Pass}> = ({pose, side, pass}) => {
  const [thigh, shin] = side === 'L' ? pose.legL : pose.legR;
  const hx = side === 'L' ? -HIP_X : HIP_X;
  // The ankle counter-rotates by the whole chain so the sole stays level with
  // the ground instead of pivoting with the shin — the single tell that
  // separates a walk from a pair of scissors opening and closing.
  const ankle = -(thigh + shin);
  const far = side === 'R';
  return (
    <g transform={`translate(${hx},${HIP_Y}) rotate(${thigh})`}>
      <Bone len={THIGH} pass={pass} far={far} />
      <g transform={`translate(0,${THIGH}) rotate(${shin})`}>
        <Bone len={SHIN} pass={pass} far={far} />
        <g transform={`translate(0,${SHIN}) rotate(${ankle})`}>
          <Shoe flip={side === 'R'} pass={pass} far={far} />
        </g>
      </g>
    </g>
  );
};

// ── the note: paper, not cardboard ──────────────────────────────────────────
/** Both long edges take the SAME travelling wave, so the note keeps its
 *  thickness and bends like a strip of paper carried through air. A rectangle
 *  that only translates reads as a sticker sliding across the screen; the
 *  ripple is most of what makes it read as money instead. */
const notePath = (wave: number, amp: number) => {
  const a = amp * Math.sin(wave);
  const b = amp * Math.sin(wave + Math.PI);
  const k = amp * 1.25;
  const L = -HALF_W;
  const R = HALF_W;
  const T = -HALF_H;
  const B = HALF_H;
  const c1 = L + NOTE_W * 0.33;
  const c2 = L + NOTE_W * 0.67;
  return (
    `M${L},${T + a} C${c1},${T + a - k} ${c2},${T + b + k} ${R},${T + b}` +
    ` L${R},${B + b} C${c2},${B + b + k} ${c1},${B + a - k} ${L},${B + a} Z`
  );
};

const innerPath = (wave: number, amp: number, inset: number) => {
  const a = amp * Math.sin(wave);
  const b = amp * Math.sin(wave + Math.PI);
  const k = amp * 1.25;
  const L = -HALF_W + inset;
  const R = HALF_W - inset;
  const T = -HALF_H + inset;
  const B = HALF_H - inset;
  const c1 = L + (R - L) * 0.33;
  const c2 = L + (R - L) * 0.67;
  return (
    `M${L},${T + a} C${c1},${T + a - k} ${c2},${T + b + k} ${R},${T + b}` +
    ` L${R},${B + b} C${c2},${B + b + k} ${c1},${B + a - k} ${L},${B + a} Z`
  );
};

const Foot: React.FC<{fill: string}> = ({fill}) => (
  <g fill={fill}>
    <ellipse cx={0} cy={2.5} rx={4.6} ry={6.6} />
    <circle cx={-3.6} cy={-5.6} r={1.7} />
    <circle cx={0.6} cy={-7.4} r={1.8} />
    <circle cx={4.4} cy={-5.2} r={1.6} />
  </g>
);

const Brows: React.FC<{brow: Pose['brow']}> = ({brow}) => {
  const s = {fill: 'none', stroke: OUTLINE, strokeWidth: 4.5, strokeLinecap: 'round' as const};
  if (brow === 'raised') return (<g {...s}><path d="M-29,-32 q10,-8 20,-4" /><path d="M9,-30 q10,-6 20,-2" /></g>);
  if (brow === 'worried') return (<g {...s}><path d="M-29,-22 q10,-6 20,2" /><path d="M9,-20 q10,-8 20,-2" /></g>);
  if (brow === 'delighted') return (<g {...s}><path d="M-29,-30 q10,-7 20,-1" /><path d="M9,-31 q10,-6 20,1" /></g>);
  if (brow === 'level') return (<g {...s}><path d="M-29,-26 h20" /><path d="M9,-26 h20" /></g>);
  // curious: one brow up, the magnifying-glass side. The default, because the
  // character's whole job is looking into where the money went.
  return (<g {...s}><path d="M-29,-30 q10,-6 20,-2" /><path d="M9,-25 q10,-4 20,0" /></g>);
};

const Mouth: React.FC<{mouth: Pose['mouth']}> = ({mouth}) => {
  const s = {fill: 'none', stroke: OUTLINE, strokeWidth: 4.5, strokeLinecap: 'round' as const};
  if (mouth === 'flat') return <path {...s} d="M-16,26 h32" />;
  if (mouth === 'open') return <ellipse cx={0} cy={28} rx={11} ry={14} fill={OUTLINE} />;
  if (mouth === 'grin') return <path {...s} strokeWidth={5} d="M-20,20 q20,22 40,-4" />;
  return <path {...s} d="M-18,26 q18,14 36,-2" />;
};

// ── the whole character ─────────────────────────────────────────────────────
export const Mascot: React.FC<{pose: Pose; scale?: number; facing?: 1 | -1}> = ({
  pose, scale = 1, facing = 1,
}) => {
  const p = pose;
  // Anything inside the note shears with the paper, so the face does not pop
  // out of a bending body.
  const shear = (p.waveAmp * 2 * Math.sin(p.wave) / NOTE_W) * 57.3 * 0.55;
  const eyeShut = 1 - 0.88 * p.blink;

  return (
    <g transform={`translate(${p.x},${p.y}) scale(${facing * scale},${scale}) rotate(${p.rot}) scale(${1 / p.squash},${p.squash})`}>
      {/* limbs live behind the note, which is what lets them emerge from it */}
      <g>
        <Leg pose={p} side="R" pass="outline" />
        <Leg pose={p} side="L" pass="outline" />
        <Arm pose={p} side="R" pass="outline" />
        {p.armFront ? null : <Arm pose={p} side="L" pass="outline" />}
        <Leg pose={p} side="R" pass="fill" />
        <Leg pose={p} side="L" pass="fill" />
        <Arm pose={p} side="R" pass="fill" />
        {p.armFront ? null : <Arm pose={p} side="L" pass="fill" />}
      </g>

      {/* the note = the body */}
      <path d={notePath(p.wave, p.waveAmp)} fill={GREEN} stroke={OUTLINE} strokeWidth={5} strokeLinejoin="round" />
      <g transform={`skewY(${shear})`}>
        <path d={innerPath(p.wave, p.waveAmp, 14)} fill="none" stroke={DEEP} strokeWidth={2.5} />
        <path d={innerPath(p.wave, p.waveAmp, 22)} fill="none" stroke={DEEP} strokeWidth={2.5} opacity={0.55} />

        {/* wordless guilloche. No lettering anywhere, in any alphabet: it is
            what the style rules demand of every surface in frame, and it is
            also what keeps this from reading as any country's banknote. */}
        <g fill="none" stroke={DEEP} strokeWidth={2.5} strokeLinecap="round" opacity={0.7}>
          <path d="M-106,-24 q14,-12 28,0 q14,12 28,0" />
          <path d="M-106,-10 q14,-12 28,0 q14,12 28,0" />
          <path d="M50,-24 q14,-12 28,0 q14,12 28,0" />
          <path d="M50,-10 q14,-12 28,0 q14,12 28,0" />
          <path d="M-104,26 h56 M-104,36 h44 M48,26 h56 M60,36 h44" />
        </g>

        {/* the fixed symbol: a footprint. Ours, tied to no country and no era,
            and the same shape as the trail the character leaves behind it. */}
        {[[-106, -52], [106, -52], [-106, 52], [106, 52]].map(([cx, cy], i) => (
          <g key={i}>
            <circle cx={cx} cy={cy} r={13} fill={PALE} stroke={OUTLINE} strokeWidth={3.5} />
            <g transform={`translate(${cx},${cy}) scale(0.9)`}>
              <Foot fill={DEEP} />
            </g>
          </g>
        ))}

        {/* THE FACE, inside the portrait frame. No second head. */}
        <ellipse cx={0} cy={0} rx={56} ry={66} fill={PALE} stroke={OUTLINE} strokeWidth={5} />
        <ellipse cx={0} cy={0} rx={49} ry={59} fill="none" stroke={DEEP} strokeWidth={2.5} opacity={0.8} />
        <g fill={OUTLINE} transform={`translate(${p.look[0]},${p.look[1]})`}>
          <ellipse cx={-18} cy={-8} rx={6} ry={7.5 * eyeShut} />
          <ellipse cx={19} cy={-8} rx={6} ry={7.5 * eyeShut} />
        </g>
        <Brows brow={p.brow} />
        <Mouth mouth={p.mouth} />
      </g>

      {/* the history tell: a vintage bowler. It dates the character without
          dating the money, and it pairs with the glass — a Victorian clerk who
          investigates where the money went. */}
      <g transform={`translate(0,${-HALF_H + p.hatLift}) rotate(${p.hatRot})`}>
        <path d="M-44,0 C-46,-32 -32,-56 0,-56 C32,-56 46,-32 44,0 Z" fill={HAT} stroke={OUTLINE} strokeWidth={5} strokeLinejoin="round" />
        <path d="M-45,-12 C-24,-20 24,-20 45,-12 L45,0 L-45,0 Z" fill={HATBAND} />
        <ellipse cx={0} cy={0} rx={63} ry={12} fill={HAT} stroke={OUTLINE} strokeWidth={5} />
      </g>

      {p.armFront ? (
        <g>
          <Arm pose={p} side="L" pass="outline" />
          <Arm pose={p} side="L" pass="fill" />
        </g>
      ) : null}

      {/* the magnifying glass, in the hand that the left arm's chain ends at */}
      <MagnifyingGlass pose={p} />
    </g>
  );
};

/** The glass continues the forearm's direction rather than sitting in its own
 *  frame: the hand chain is already rotated most of the way round, so drawing
 *  the glass "up" in hand-local coordinates points it at the floor. */
const MagnifyingGlass: React.FC<{pose: Pose}> = ({pose}) => {
  const [upper, fore] = pose.armL;
  // A lens held over an eye has to MAGNIFY the eye, or it is not a lens, it is
  // a hoop. The contents are counter-rotated out of the arm's frame so the eye
  // inside stays upright however the arm is holding it.
  const upright = -(upper + fore + pose.glassRot);
  return (
    <g transform={`translate(${-SHOULDER_X},${SHOULDER_Y}) rotate(${upper}) translate(0,${UPPER_ARM}) rotate(${fore}) translate(0,${FOREARM}) rotate(${pose.glassRot})`}>
      <line x1={0} y1={2} x2={0} y2={26} stroke={OUTLINE} strokeWidth={18} strokeLinecap="round" />
      <line x1={0} y1={2} x2={0} y2={26} stroke={WOOD} strokeWidth={10} strokeLinecap="round" />
      <circle cx={0} cy={54} r={35} fill={pose.magnify ? PALE : '#DCEEF6'} fillOpacity={pose.magnify ? 1 : 0.92} stroke={OUTLINE} strokeWidth={5} />
      {pose.magnify ? (
        <g transform={`translate(0,54) rotate(${upright})`}>
          <ellipse cx={-1} cy={-4} rx={11} ry={14 * (1 - 0.88 * pose.blink)} fill={OUTLINE} />
          <path d="M-14,-26 q10,-6 20,-2" fill="none" stroke={OUTLINE} strokeWidth={5} strokeLinecap="round" />
        </g>
      ) : null}
      <circle cx={0} cy={54} r={35} fill="none" stroke={BRASS} strokeWidth={7} />
      <path d="M19,70 q-11,12 -26,9" fill="none" stroke="#FFFFFF" strokeWidth={6} strokeLinecap="round" opacity={0.9} />
    </g>
  );
};

export const Footprint: React.FC<{x: number; y: number; rot: number; scale: number; opacity: number; fill?: string}> = ({
  x, y, rot, scale, opacity, fill = TRAIL_NEAR,
}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${scale})`} opacity={opacity}>
    <Foot fill={fill} />
  </g>
);
