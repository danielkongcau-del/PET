# PET realtime motion generator

This process receives versioned `pet-motion` NDJSON on stdin and writes only
protocol NDJSON on stdout. Diagnostics always go to stderr. It currently uses a
small stochastic autoregressive planner; it does **not** replay animation clips.
The planner is the dependency-free vertical-slice backend and will later be
replaced by a learned PyTorch backend behind `MotionBackend`.

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
