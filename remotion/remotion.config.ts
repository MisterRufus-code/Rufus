import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
// CPU-friendly: x264 at CRF 20 matches the FFmpeg pipeline's quality target.
Config.setCodec('h264');
Config.setCrf(20);
