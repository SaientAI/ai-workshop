# Blender Relief Asset Pipeline

This pipeline turns PNG art into prototype `.glb` relief assets without using any AI generation tokens.

It is not a true image-to-3D generator. A single PNG cannot provide accurate side anatomy, hidden surfaces, rigging, or game-ready topology. For production 3D assets, export real `.glb` files from Meshy, Blender, Tripo, or another 3D tool directly into `assets/game-assets/`.

## Folders

- `assets/source-png/` - PNGs for relief/prototype conversion.
- `assets/game-assets/` - production `.glb` files and converted relief `.glb` files.

## Commands

```bash
npm run asset:test
npm run asset:dry-run
npm run asset:build
```

`asset:test` always exits. If Blender is not installed, it validates the Python side and reports that real conversion was skipped.

`asset:dry-run` scans `assets/source-png/` and prints the conversion plan without running Blender.

`asset:build` runs Blender in background mode once per PNG and writes relief `.glb` files.

## Blender Setup

Install Blender and make sure `blender` is on `PATH`, or point the wrapper at it:

```bash
BLENDER_BIN=/path/to/blender npm run asset:build
```

The generated GLB is an alpha cutout relief mesh with depth. It is useful for placeholders, icons, distant props, and quick tile tests. It is not equivalent to a sculpted 3D character or building.
