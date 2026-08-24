import React from 'react';
import {Composition} from 'remotion';
import {Short, ShortProps, FPS, WIDTH, HEIGHT} from './Short';
import {MascotShowreel, MascotWalk, MascotIdle, MascotInspect, MascotEntrance} from './MascotScenes';

const defaultProps: ShortProps = {
  job: '',
  clips: [],
  voice: '',
  music: null,
  words: [],
  durationInSeconds: 30,
  inserts: [],
};

export const Root: React.FC = () => {
  return (
    <>
    <Composition
      id="RufusShort"
      component={Short}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={30 * FPS}
      defaultProps={defaultProps}
      calculateMetadata={({props}) => ({
        durationInFrames: Math.max(1, Math.ceil(props.durationInSeconds * FPS)),
        // The composition takes its SHAPE from the run, so one component
        // renders both formats. Without this a long-form job rendered at the
        // vertical default and every landscape frame came out cropped.
        width: props.width && props.width > 0 ? Math.round(props.width) : WIDTH,
        height: props.height && props.height > 0 ? Math.round(props.height) : HEIGHT,
      })}
    />
    {/* The mascot renders on its own so it can be reviewed and iterated on
        without waiting for a full job — and so a scene can be dropped into a
        video as a finished clip rather than rebuilt each time. */}
    <Composition id="MascotShowreel" component={MascotShowreel} fps={FPS} width={WIDTH} height={HEIGHT} durationInFrames={Math.round(15.2 * FPS)} />
    <Composition id="MascotWalk" component={MascotWalk} fps={FPS} width={WIDTH} height={HEIGHT} durationInFrames={Math.round(6 * FPS)} />
    <Composition id="MascotIdle" component={MascotIdle} fps={FPS} width={WIDTH} height={HEIGHT} durationInFrames={Math.round(5 * FPS)} />
    <Composition id="MascotInspect" component={MascotInspect} fps={FPS} width={WIDTH} height={HEIGHT} durationInFrames={Math.round(4 * FPS)} />
    <Composition id="MascotEntrance" component={MascotEntrance} fps={FPS} width={WIDTH} height={HEIGHT} durationInFrames={Math.round(4 * FPS)} />
    </>
  );
};
