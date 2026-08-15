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
      })}
    />
  );
};
