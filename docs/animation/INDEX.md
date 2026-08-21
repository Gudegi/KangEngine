# Animation

KangEngine can load BVH, FBX, and AMASS motion, evaluates skeleton poses, displays
skeletons or skinned characters, and provides a motion sequencer panel.

- [Load and Play Motion](LOAD_MOTION.md)
- [Pose Visualization](VISUALS.md)
- [Motion Retargeting](RETARGETING.md)
- [SMPL Models](SMPL.md)

All supported motion sources produce the same `ke.animation.SkeletonMotion`
type, so playback, sampling, analysis, visualization, and export do not depend
on the source file format.
