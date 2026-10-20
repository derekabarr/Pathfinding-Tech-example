# Phaser 4 / WebGL2 GLSL pathfinding

Run from a **local web server**, not a `file://` URL:

```bash
cd GLSL_Pathfinding
python3 -m http.server 8000
```

Open <http://localhost:8000/>.

## Controls

- **Click** a light/passable region to set a new goal (distance textures clear on the GPU).
- **[ / ]** decrease/increase the number of full-grid wavefront passes per frame.

## Design

- Phaser **4.2.1** creates a WebGL2 rendering context, shared with a Phaser 4 `Extern` game object.
- The canvas maze is generated locally in `generateMazeCanvas()` and uploaded to GPU **once** (`texImage2D` with that canvas). The GPU passability test reads the **actual maze texture pixels**.
- The original CPU canvas pixels are inspected *once* for initial agent spawning and goal-click validation; **they are not uploaded separately** as a mask.
- Initial positions for **1,000 agents** are a separate one-time **32×32 RGBA8 (~4 KiB)** upload.
- Two 1024×1024 GPU RGBA8 ping-pong textures store the path-distance field. No CPU pixel array is uploaded for them; `texImage2D(..., null)` allocates VRAM.
- A shader samples the current cell's **four** orthogonal neighbors each pass (GLSL equivalent of a 3×3 cross convolution). An unvisited cell becomes `1 + min(positive neighbor distance)`, with the goal at 1.
- Distances are exact 16-bit integers **packed into RG** of RGBA8 textures. This removes a float-framebuffer-extension requirement. `0` means unvisited. The maximum value is `65535`; propagation stops after `min(size², 65534)` passes to prevent overflow and the HUD labels it *PARTIAL* if the cap is hit.
- Another shader updates all agent positions using the distance texture. The **agent positions never move through the CPU** after initialization.
- Agents are drawn as GL point primitives using a vertex shader fetching each agent position from GPU memory.
- Wavefront completion uses WebGL2's `ANY_SAMPLES_PASSED` occlusion query every 64 passes, polled with `QUERY_RESULT_AVAILABLE`; it returns only a boolean scalar, **never** a full texture or `readPixels`.
- When the goal changes, the distance buffers are cleared on GPU. Maze and agent-position textures are **not reuploaded**.
- Direct GL commands run inside Phaser 4's supported `Extern.render()`, which restores Phaser rendering state after an external draw. There are **no Phaser 3 `renderer.pipelines` calls**.
- WebGL context loss is detected; after restoration, GPU resources are recreated from saved CPU initialization data. This is the exceptional case that requires uploading the maze again.

## Technical tradeoffs

- This is wavefront/BFS (uniform-cost 4-neighbor shortest paths), **not a neural network**.
- The GPU does one full texture pass per expansion step: a 1024×1024 maze can still require hundreds/thousands of passes. Tune `stepsPerFrame` according to hardware; more passes means quicker convergence but potentially lower FPS.
- Position updates are one pixel/cell per frame; they wait until a distance becomes available.
- For accurate GL integer texture encoding, GPU dithering/blending is off during compute.
- The path field has 16-bit maximum cost; extremely long snake paths can exceed the cap (HUD reports partial).
- `Extern.render()` targets the full 1024×1024 canvas and assumes one full-screen camera; resize/multi-camera needs additional camera viewport transforms.
- The maze image data is not transferred to another GPU context; **all work shares Phaser's WebGL2 context**.

## Files

- `index.html` — game initialization and WebGL2 context.
- `src/GameScene.js` — Phaser scene, goal click input, HUD.
- `src/GpuPathfindingVisualizer.js` — GPU resources, ping-pong passes, GPU agents, Extern render.
- `shaders/pathfinding.glsl` — wavefront convolution, progress, agent update and visualization fragments.

## Caveat

This version has been statically checked, but has not yet been tested against the Phaser 4.2.1 CDN at runtime in the target browser. If your Phaser build has different `Extern` behavior, check the browser console.
