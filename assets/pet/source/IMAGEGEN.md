# Generated concept provenance

The concept master was created with the built-in image generation tool using
`cat-reference.png` as the identity and silhouette reference. The generated
source used a green chroma background. Image generation itself is not an
offline deterministic step; the prompt below records its provenance. From the
checked-in `cat-master-chroma.png` onward, the build is reproducible:

```powershell
& 'D:\Anaconda\envs\pet-core\python.exe' tools\assets\remove_chroma_key.py `
  --source assets\pet\source\cat-master-chroma.png `
  --output assets\pet\runtime\cat-master-transparent.png

& 'D:\Anaconda\envs\pet-core\python.exe' tools\assets\build_cat_assets.py `
  --source assets\pet\runtime\cat-master-transparent.png `
  --output assets\pet\runtime\cat-48.png `
  --metadata assets\pet\runtime\cat-parts.json
```

The first command estimates the chroma colour from the noisy image border,
creates a feathered alpha edge, and removes green spill. The second command
performs the deterministic 48x48 palette reduction and writes part metadata.

## Prompt

```text
Use case: stylized-concept
Asset type: Windows desktop pet pixel-art master sprite
Primary request: Recreate the cat from Image 1 as a clean, production-ready pixel-art sprite. Preserve its unmistakable silhouette: compact white cat, black one-pixel-style outline, two pointed ears, tiny legs, rounded rectangular body, and curled upright tail on the right. Keep the same cute minimal expression and proportions.
Input images: Image 1 is the identity and silhouette reference.
Scene/backdrop: perfectly flat solid #00ff00 chroma-key background for background removal; one uniform color, no floor, shadows, gradients, texture, reflections, or lighting variation.
Style/medium: strict low-resolution 48x48 pixel art, hard square pixels, nearest-neighbor look, no antialiasing, no soft edges.
Composition/framing: exactly one side-view cat centered with generous green padding; all parts fully visible.
Color palette: black outline, white/off-white body, minimal pale-pink inner ears only if already implied; do not use #00ff00 in the cat.
Constraints: no text, no watermark, no accessories, no extra objects, no cast shadow, no contact shadow, no reflection. Preserve the reference design rather than inventing a new cat.
```
