# PET realtime motion generator

This process receives versioned `pet-motion` NDJSON on stdin and writes only
protocol NDJSON on stdout. Diagnostics always go to stderr. It currently uses a
small stochastic autoregressive planner; it does **not** replay animation clips.
The planner is the dependency-free vertical-slice backend and will later be
replaced by a learned PyTorch backend behind `MotionBackend`.

The offline `pet_generator.data_gen` command is deliberately different: it
combines procedural desktop/root trajectories with authored clips declared by
the selected character manifest, converting absolute glTF local rotations to
rest-local quaternion deltas. Runtime generation still never plays those clips.

Example (from `services/generator`):

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' -m pet_generator.data_gen `
  --episodes 1000 --duration-ms 30000 --output ..\..\data\train\cat-v1
```

Use a fresh output directory for every run. Its `dataset-manifest.json` binds
the exact feature/target order, K/H/dt, character rig, authored clip
fingerprints and the destination checkpoint ABI; episode files from different
characters or rig revisions are never mixed.

The two runtime clocks are intentionally separate. `--world-state-dt-ms`
defaults to 50 ms, matching the Electron host's 20 Hz `world_state` publisher,
while `--dt-ms` defaults to 33 ms and controls plan keyframes. Between two
world states the simulator samples the active plan from `generated_at_ms` with
linear wall-clock interpolation on one long-lived deterministic 16 ms motion
timer. Its absolute phase continues across 50 ms replans; the simulator never
invents a short callback at a world-state boundary. Both cadences, timer phase,
and the sampling rule are recorded in `dataset-manifest.json`.

## Run

From the repository root (the command used by the Electron host):

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' services/generator/run.py
```

Or from this directory:

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' -m pet_generator
```

`--seed 7` (or `PET_GENERATOR_SEED=7`) makes plans reproducible. Without an
explicit seed, each process starts with a random session seed. Every
`horizon_plan` reports the actual uint32 seed used for that state so a session
can be replayed exactly.

The default backend is explicitly `procedural`. `--backend neural` /
`PET_GENERATOR_BACKEND=neural` currently fails closed because the neural model
loader has not been implemented. Likewise, setting `PET_GENERATOR_CHECKPOINT`
while using the procedural backend is an error; a configured checkpoint is
never silently ignored and presented as active inference.

## Tests

No pytest or runtime package installation is required:

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' -m unittest discover `
  -s services/generator/tests -t services/generator -v
```

Manual protocol smoke test:

```powershell
Get-Content services/generator/examples/smoke_input.ndjson |
  & 'D:\Anaconda\envs\pet-core\python.exe' services/generator/run.py --seed 7 --metrics-interval-ms 0
```

The expected output types are `ready`, `horizon_plan`, and `pong`. The log lines
shown by the terminal are emitted on stderr and are never mixed into the pipe.

## Runtime semantics

- Coordinates are Windows physical screen pixels with a top-left origin.
- Plan `dx`/`dy` values are relative to the triggering world's foot anchor.
- Only the newest pending `world_state` is retained. When replacement happens,
  unconsumed click edge events are merged by id instead of silently discarded.
- A click targeted at the pet has priority over autonomous locomotion.
- User drag, hidden/fullscreen state, malformed input, and backend failure are
  handled without giving the generator authority over real window positions.
- The Electron host remains the single writer and may reject or clamp any point.
- `cancel` is one-way in protocol v1. Closing stdin cleanly stops the service.

## Learned backend seam

Implement `MotionBackend.generate(world, seed, generated_at_ms)` and return a
`MotionPlan`. A future torch backend should subclass `TorchMotionBackend`, import
torch lazily in its constructor, and perform warmup before reporting `ready`.
Neither the stdio service nor the desktop host needs to change.

The implementation is character-agnostic, but checkpoints are not shared
between characters. A future learned backend must load the selected manifest
and reject a checkpoint unless its `characterId`, `rigFingerprint`, exact
`drivenJointOrder`, dataset schema and normalization metadata all match.
The canonical artifact is a `pet-character-motion-checkpoint-v1` bundle at
`checkpoints/characters/<characterId>/<rigFingerprint>/motion.pt`, with
`pet-character-motion-checkpoint-metadata-v1` metadata. Distinct characters
must be trained and stored separately even when their rigs are byte-identical.
