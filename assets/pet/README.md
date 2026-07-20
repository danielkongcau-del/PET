# PET cat assets

- `source/cat-reference.png` is the immutable screenshot supplied by the user.
- `source/cat-master-chroma.png` is the image-generated high-resolution concept
  on a flat chroma background.
- `runtime/cat-master-transparent.png` is the alpha-validated concept master.
- `runtime/cat-48.png` is the deterministic 48x48 runtime sprite derived from
  the generated concept master and used at 2x nearest-neighbour scale.
- `runtime/cat-parts.json` defines the logical foot anchor and records coarse
  legacy part clips. It also records that this master faces left. The current
  renderer intentionally draws the flattened
  sprite in one call: rectangular clips must not drive independent transforms,
  because they expose seams. Independent limbs require genuine overlapping
  RGBA semantic layers in a future asset revision.

The image-generation step is recorded in `source/IMAGEGEN.md` and is not an
offline deterministic build step. Starting from the checked-in image-generated
chroma master, rebuild both deterministic derivatives with:

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' tools\assets\remove_chroma_key.py `
  --source assets\pet\source\cat-master-chroma.png `
  --output assets\pet\runtime\cat-master-transparent.png

& 'D:\Anaconda\envs\pet-core\python.exe' tools\assets\build_cat_assets.py `
  --source assets\pet\runtime\cat-master-transparent.png `
  --output assets\pet\runtime\cat-48.png `
  --metadata assets\pet\runtime\cat-parts.json
```

The keyer estimates the noisy green-screen colour from the image border,
removes pixels close to that colour, feathers mixed edge pixels, and suppresses
green spill in partially transparent pixels. Its defaults are calibrated for
the checked-in concept master and can be overridden from `--help`.

Verify the complete deterministic asset path with:

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' -m unittest discover `
  -s tools/assets -p test_*.py -v
```
