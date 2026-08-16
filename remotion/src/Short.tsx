import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  Loop,
  OffthreadVideo,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {continueRender, delayRender} from 'remotion';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';

// Self-hosted Anton (staged into public/fonts/ by remotion_renderer.py) — no
// network dependency at render time. Falls back to Arial Black if missing.
const fontFamily = 'Anton, "Arial Black", Arial, sans-serif';
if (typeof document !== 'undefined') {
  const handle = delayRender('load Anton font');
  const font = new FontFace('Anton', `url(${staticFile('fonts/Anton-Regular.ttf')})`);
  font
    .load()
    .then(() => {
      document.fonts.add(font);
      continueRender(handle);
    })
    .catch(() => continueRender(handle));
}

export const FPS = 30;
// The DEFAULT shape, not the only one. Python's video_format profile decides
// what a run actually renders at (1080x1920 for a Short, 1920x1080 for
// long-form) and passes it in props; Root's calculateMetadata applies it. These
// stay as the fallback for a props-less preview in the Remotion studio.
export const WIDTH = 1080;
export const HEIGHT = 1920;

const XFADE_SEC = 0.35; // crossfade between clips
const EDGE_FADE_SEC = 0.45; // fade from/to black at the edges
const MUSIC_VOL = 0.14;

export type Word = {text: string; start: number; end: number};

// What the edit director decided for one beat. Optional throughout: a run
// without a plan renders exactly as it always did.
export type BeatDirection = {
  n: number;
  motion: 'push_in' | 'pull_back' | 'hold_still' | 'drift_left' | 'drift_right' | 'rise';
  intensity: 'subtle' | 'normal' | 'strong';
  emphasis: string[];
};

export type EditPlan = {
  peak_beat: number;
  beats: BeatDirection[];
};

// One word-synced cutaway. `at` is the second the narrator SAYS the word, taken
// from the same Whisper pass that drives the captions — see insert_director.py.
export type Insert = {
  word: string;
  at: number;
  hold: number;
  file: string; // filename inside public/<job>/
};

export type ShortProps = {
  job: string;
  clips: string[]; // filenames inside public/<job>/
  clipDurations?: (number | null)[]; // seconds per clip (for looping short sources)
  voice: string; // voice mp3 filename inside public/<job>/
  music: string | null; // optional music filename inside public/<job>/
  words: Word[];
  durationInSeconds: number;
  edit?: EditPlan | null; // per-beat direction; null = use the default cycle
  inserts?: Insert[] | null; // word-synced cutaways; absent = the old look
  width?: number; // from video_format; absent = the vertical default
  height?: number;
};

const HIGHLIGHT = /[\d$%]/;

// ── Word-synced insert ───────────────────────────────────────────────────────
// The format this serves: the narrator says "palace" and a palace appears on
// the word. Not a transition — the scene underneath does not change, an object
// lands on top of it and leaves. That is why it pops rather than fades, and why
// the sound under it is a blip and not a whoosh.
const INSERT_FRACTION = 0.42; // of frame width — big enough to read, small
                              // enough that the beat behind it still reads
const InsertLayer: React.FC<{inserts: Insert[]; job: string}> = ({inserts, job}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();

  return (
    <>
      {inserts.map((ins, i) => {
        const startFrame = Math.round(ins.at * fps);
        const holdFrames = Math.max(1, Math.round((ins.hold || 0.7) * fps));
        const local = frame - startFrame;
        if (local < 0 || local > holdFrames) return null;

        // Spring in, hard cut out. An insert that fades away competes for
        // attention with the next word; one that simply stops does not.
        const scale = spring({
          frame: local,
          fps,
          config: {damping: 12, mass: 0.5, stiffness: 190},
          from: 0.55,
          to: 1,
        });
        // Alternate sides so twenty of them do not stack in one place, and
        // keep them clear of the caption band at the vertical centre.
        const left = i % 2 === 0;
        const size = width * INSERT_FRACTION;

        return (
          <AbsoluteFill key={`ins-${i}-${ins.word}`} style={{pointerEvents: 'none'}}>
            <div
              style={{
                position: 'absolute',
                width: size,
                height: size,
                left: left ? width * 0.06 : width - size - width * 0.06,
                top: i % 4 < 2 ? height * 0.16 : height * 0.62,
                transform: `scale(${scale}) rotate(${left ? -3 : 3}deg)`,
                borderRadius: 18,
                overflow: 'hidden',
                boxShadow: '0 18px 48px rgba(0,0,0,0.55)',
                border: '6px solid #fff',
                backgroundColor: '#fff',
              }}
            >
              <Img
                src={staticFile(`${job}/${ins.file}`)}
                style={{width: '100%', height: '100%', objectFit: 'cover'}}
              />
            </div>
          </AbsoluteFill>
        );
      })}
    </>
  );
};

// ── One background clip with Ken Burns motion ────────────────────────────────
// The director's vocabulary, expressed as the motion this component already
// speaks. Kept beside the fallback cycle on purpose: both must stay in the
// same units, or a directed beat would move at a different scale from an
// undirected one in the same video.
const MOTION_PATTERNS: Record<string, {zoomIn: boolean; dx: number; dy: number}> = {
  push_in:     {zoomIn: true,  dx: 0,   dy: 0},
  pull_back:   {zoomIn: false, dx: 0,   dy: 0},
  drift_left:  {zoomIn: true,  dx: -26, dy: 0},
  drift_right: {zoomIn: true,  dx: 26,  dy: 0},
  rise:        {zoomIn: true,  dx: 0,   dy: -24},
  // hold_still is handled separately: it is the ABSENCE of motion, and the
  // director's most valuable instruction. A number or a reveal lands hardest
  // on a frame that does not move, and a video where every beat drifts has no
  // emphasis anywhere. A hair of scale keeps it from reading as a freeze.
  hold_still:  {zoomIn: true,  dx: 0,   dy: 0},
};

const INTENSITY_SCALE: Record<string, number> = {
  subtle: 0.5,
  normal: 1,
  strong: 1.6,
};

const KenBurnsClip: React.FC<{
  src: string;
  index: number;
  clipFrames: number;
  sourceDurationSec?: number | null;
  direction?: BeatDirection | null;
}> = ({src, index, clipFrames, sourceDurationSec, direction}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = Math.min(1, frame / Math.max(1, clipFrames));
  // 6 zoom+drift patterns (vs the old %2/%3 combo that repeated every 6 clips
  // with only 3 distinct drifts) — consecutive clips always move differently.
  // This is now the FALLBACK: the same cycle, in the same order, for every
  // video ever rendered, which is what the director exists to replace.
  const KB_PATTERNS = [
    {zoomIn: true,  dx: -22, dy: 12},  // push in, drift left-down
    {zoomIn: false, dx: 22,  dy: -16}, // pull back, drift right-up
    {zoomIn: true,  dx: 18,  dy: 18},  // push in, drift right-down
    {zoomIn: false, dx: -16, dy: -10}, // pull back, drift left-up
    {zoomIn: true,  dx: 0,   dy: -24}, // push in, rise
    {zoomIn: false, dx: -26, dy: 0},   // pull back, slide left
  ];
  const directed = direction ? MOTION_PATTERNS[direction.motion] : undefined;
  const base = directed ?? KB_PATTERNS[index % KB_PATTERNS.length];
  const still = direction?.motion === 'hold_still';
  const k = still ? 0.12 : INTENSITY_SCALE[direction?.intensity ?? 'normal'] ?? 1;
  const p = {zoomIn: base.zoomIn, dx: base.dx * k, dy: base.dy * k};
  // 0.12 of the usual travel on a hold_still, 1.6x on a strong push. The base
  // 1.04 -> 1.16 stays the reference so a directed beat and an undirected one
  // in the same video move at comparable scale.
  const zoomTravel = 0.12 * k;
  const scale = p.zoomIn
    ? interpolate(t, [0, 1], [1.04, 1.04 + zoomTravel])
    : interpolate(t, [0, 1], [1.04 + zoomTravel, 1.04]);
  const driftX = interpolate(t, [0, 1], [0, p.dx]);
  const driftY = interpolate(t, [0, 1], [0, p.dy]);

  const video = (
    <OffthreadVideo
      src={src}
      muted
      style={{
        width: '100%',
        height: '100%',
        objectFit: 'cover',
        transform: `scale(${scale}) translate(${driftX}px, ${driftY}px)`,
      }}
    />
  );

  // Loop sources shorter than their on-screen slot so playback never freezes.
  const sourceFrames = sourceDurationSec
    ? Math.max(1, Math.floor(sourceDurationSec * fps))
    : null;
  const needsLoop = sourceFrames !== null && sourceFrames < clipFrames;

  return (
    <AbsoluteFill style={{overflow: 'hidden', backgroundColor: '#000'}}>
      {needsLoop ? <Loop durationInFrames={sourceFrames!}>{video}</Loop> : video}
      {/* Subtle cinematic grade: vignette */}
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,0.45) 100%)',
        }}
      />
    </AbsoluteFill>
  );
};

// ── Animated word caption (Hormozi style: one word, spring pop) ──────────────
const Captions: React.FC<{words: Word[]}> = ({words}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;

  // SIZED FROM THE FRAME, not from 1080x1920. fontSize 96 and paddingBottom
  // 700 were right for a phone: big words, lifted clear of the Shorts UI that
  // covers the bottom fifth. On a 1080-tall landscape frame the same two
  // numbers are a caption floating 65% up the picture. The portrait ratios
  // below reproduce 96 and 700 exactly, so nothing about the existing channel
  // moves; landscape gets broadcast proportions instead — smaller, and near
  // the bottom edge where no app UI has to be avoided.
  const portrait = height >= width;
  const fontSize = Math.round(height * (portrait ? 0.05 : 0.055));
  const paddingBottom = Math.round(height * (portrait ? 0.3646 : 0.07));

  const active = words.find((w) => t >= w.start && t < w.end);
  if (!active) return null;

  const startFrame = Math.round(active.start * fps);
  const pop = spring({
    frame: frame - startFrame,
    fps,
    config: {damping: 11, stiffness: 240, mass: 0.6},
    durationInFrames: 8,
  });
  const scale = interpolate(pop, [0, 1], [1.32, 1]);
  const color = HIGHLIGHT.test(active.text) ? '#00FF44' : '#FFFFFF';

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        paddingBottom,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          fontFamily,
          fontSize,
          fontWeight: 900,
          color,
          textTransform: 'uppercase',
          transform: `scale(${scale})`,
          textShadow:
            '0 0 18px rgba(0,0,0,0.9), 4px 4px 0 #000, -4px 4px 0 #000, 4px -4px 0 #000, -4px -4px 0 #000',
          letterSpacing: 2,
          textAlign: 'center',
          maxWidth: '88%',
          lineHeight: 1.05,
        }}
      >
        {active.text}
      </div>
    </AbsoluteFill>
  );
};

// ── Thin retention progress bar ───────────────────────────────────────────────
const ProgressBar: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const pct = (frame / Math.max(1, durationInFrames - 1)) * 100;
  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        height: 10,
        width: `${pct}%`,
        background: 'linear-gradient(90deg, #00FF44, #00C2FF)',
        borderRadius: 5,
      }}
    />
  );
};

// ── Edge fade from/to black ────────────────────────────────────────────────────
const EdgeFade: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const fadeFrames = Math.round(EDGE_FADE_SEC * fps);
  const opacity = interpolate(
    frame,
    [0, fadeFrames, durationInFrames - fadeFrames, durationInFrames - 1],
    [1, 0, 0, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  if (opacity <= 0.001) return null;
  return <AbsoluteFill style={{backgroundColor: '#000', opacity}} />;
};

// ── Main composition ───────────────────────────────────────────────────────────
export const Short: React.FC<ShortProps> = ({
  job,
  clips,
  clipDurations,
  voice,
  music,
  words,
  edit,
  inserts,
}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();

  const n = Math.max(1, clips.length);
  const transFrames = Math.round(XFADE_SEC * fps);
  // TransitionSeries: total = n*seq − (n−1)*transition → solve for seq
  const seqFrames = Math.ceil((durationInFrames + (n - 1) * transFrames) / n);

  const musicVolume = music
    ? interpolate(
        frame,
        [0, Math.round(1.5 * fps), durationInFrames - Math.round(1.5 * fps), durationInFrames - 1],
        [0, MUSIC_VOL, MUSIC_VOL, 0],
        {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
      )
    : 0;

  const asset = (name: string) => staticFile(`${job}/${name}`);

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {clips.length === 1 ? (
        <KenBurnsClip
          src={asset(clips[0])}
          index={0}
          clipFrames={durationInFrames}
          sourceDurationSec={clipDurations?.[0]}
          direction={edit?.beats?.[0]}
        />
      ) : (
        <TransitionSeries>
          {clips.flatMap((clip, i) => {
            const items = [
              <TransitionSeries.Sequence key={`seq-${i}`} durationInFrames={seqFrames}>
                <KenBurnsClip
                  src={asset(clip)}
                  index={i}
                  clipFrames={seqFrames}
                  sourceDurationSec={clipDurations?.[i]}
                  direction={edit?.beats?.[i]}
                />
              </TransitionSeries.Sequence>,
            ];
            if (i < clips.length - 1) {
              items.push(
                <TransitionSeries.Transition
                  key={`trans-${i}`}
                  presentation={fade()}
                  timing={linearTiming({durationInFrames: transFrames})}
                />,
              );
            }
            return items;
          })}
        </TransitionSeries>
      )}

      {inserts && inserts.length ? <InsertLayer inserts={inserts} job={job} /> : null}

      <Captions words={words} />
      <ProgressBar />
      <EdgeFade />

      <Audio src={asset(voice)} />
      {music ? <Audio src={asset(music)} volume={musicVolume} loop /> : null}
    </AbsoluteFill>
  );
};
