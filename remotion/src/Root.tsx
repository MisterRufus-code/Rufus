import React from 'react';
import {Composition} from 'remotion';
import {Short, ShortProps, FPS, WIDTH, HEIGHT} from './Short';

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
  );
};
