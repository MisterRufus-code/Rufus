import React from 'react';
import {
  AbsoluteFill,
  Audio,
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
export const WIDTH = 1080;
export const HEIGHT = 1920;

const XFADE_SEC = 0.35; // crossfade between clips
const EDGE_FADE_SEC = 0.45; // fade from/to black at the edges
const MUSIC_VOL = 0.14;

export type Word = {text: string; start: number; end: number};

export type ShortProps = {
  job: string;
  clips: string[]; // filenames inside public/<job>/
  clipDurations?: (number | null)[]; // seconds per clip (for looping short sources)
  voice: string; // voice mp3 filename inside public/<job>/
  music: string | null; // optional music filename inside public/<job>/
  words: Word[];
  durationInSeconds: number;
};

const HIGHLIGHT = /[\d$%]/;

// ── One background clip with Ken Burns motion ────────────────────────────────
const KenBurnsClip: React.FC<{
  src: string;
  index: number;
  clipFrames: number;
  sourceDurationSec?: number | null;
}> = ({src, index, clipFrames, sourceDurationSec}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = Math.min(1, frame / Math.max(1, clipFrames));
  // 6 zoom+drift patterns (vs the old %2/%3 combo that repeated every 6 clips
  // with only 3 distinct drifts) — consecutive clips always move differently.
  const KB_PATTERNS = [
    {zoomIn: true,  dx: -22, dy: 12},  // push in, drift left-down
    {zoomIn: false, dx: 22,  dy: -16}, // pull back, drift right-up
    {zoomIn: true,  dx: 18,  dy: 18},  // push in, drift right-down
    {zoomIn: false, dx: -16, dy: -10}, // pull back, drift left-up
    {zoomIn: true,  dx: 0,   dy: -24}, // push in, rise
    {zoomIn: false, dx: -26, dy: 0},   // pull back, slide left
  ];
  const p = KB_PATTERNS[index % KB_PATTERNS.length];
  const scale = p.zoomIn
    ? interpolate(t, [0, 1], [1.04, 1.16])
    : interpolate(t, [0, 1], [1.16, 1.04]);
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
  const {fps} = useVideoConfig();
  const t = frame / fps;

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
        paddingBottom: 700,
        pointerEvents: 'none',
      }}
    >
      <div
        style={{
          fontFamily,
          fontSize: 96,
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

      <Captions words={words} />
      <ProgressBar />
      <EdgeFade />

      <Audio src={asset(voice)} />
      {music ? <Audio src={asset(music)} volume={musicVolume} loop /> : null}
    </AbsoluteFill>
  );
};
