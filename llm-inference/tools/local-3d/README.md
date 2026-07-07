# Local Image-to-3D Pipeline

This is the local-only path for real image-to-3D generation.

It uses the official open-source TripoSR project and runs on this machine. It is separate from the Blender relief converter:

- TripoSR local 3D: attempts real single-image 3D reconstruction.
- Blender relief: creates a quick extruded/relief placeholder from a PNG.

## Setup

```bash
npm run local3d:setup
```

The setup creates `.venvs/triposr` and clones TripoSR under `tools/local-3d/vendor/TripoSR`.

## Run

Put PNGs in `assets/source-png`, then run:

```bash
npm run local3d:run
```

Generated `.glb` files are written to `assets/game-assets`.

## Notes

The first run may download model weights into the normal Hugging Face cache. That is still local generation: the model runs on this PC after the files are present.

Single-image 3D still has limits. For best results use clean object images, plain backgrounds, and orthographic/front-ish views. Multi-view generation is a separate upgrade path.
