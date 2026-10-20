/**
 * Phaser 4 + WebGL2, GPU-resident 4-neighbor wavefront and 1,000 GPU agents.
 * Uses Phaser 4's Extern to safely issue raw WebGL calls within the render loop.
 *
 * Maze canvas is uploaded ONCE with texImage2D. Positions are a separate tiny,
 * one-time upload. After initialization there are no pixel/map CPU readbacks,
 * no recurring texture uploads, and no PNG snapshots. FBO operations and
 * asynchronous occlusion queries remain on the GPU.
 */

const QUAD_VERTEX = `#version 300 es
precision highp float;
void main() {
    vec2 vertex = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(vertex * 2.0 - 1.0, 0.0, 1.0);
}`;

//const POINT_VERTEX = `#version 300 es
//precision highp float;
//precision highp int;

//uniform vec2 uSize;

//void main() {
//    int col = gl_VertexID % 32;
//    int row = gl_VertexID / 32;
//
//    float x = 32.0 + float(col) * 30.0;
//    float y = 32.0 + float(row) * 30.0;
//
//    gl_Position = vec4(
//        (x + 0.5) / uSize.x * 2.0 - 1.0,
//        1.0 - (y + 0.5) / uSize.y * 2.0,
//        0.0,
//        1.0
//    );
//    gl_PointSize = 3.0;
//}`;


const POINT_VERTEX = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uAgents;
uniform ivec2 uAgentTexSize;
uniform vec2 uSize;
void main() {
    ivec2 at = ivec2(gl_VertexID % uAgentTexSize.x, gl_VertexID / uAgentTexSize.x);
    vec4 data = texelFetch(uAgents, at, 0);
    float x = floor(data.r * 255.0 + 0.5) * 256.0 + floor(data.g * 255.0 + 0.5);
    /* float x = 32.0 + float(gl_VertexID % 32) * 30.0; */
    /* float y = 32.0 + float(gl_VertexID / 32) * 30.0; */
    float y = floor(data.b * 255.0 + 0.5) * 256.0 + floor(data.a * 255.0 + 0.5);
    gl_Position = vec4((x + 0.5) / uSize.x * 2.0 - 1.0,
                       1.0 - (y + 0.5) / uSize.y * 2.0, 0.0, 1.0);
    gl_PointSize = 3.0;
}`;


const POINT_FRAGMENT = `#version 300 es
precision highp float;
out vec4 fragColor;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    if (dot(q, q) > 1.0) discard;
    fragColor = vec4(1.0, 0.94, 0.04, 1.0);
}`;

export class GpuPathfindingVisualizer extends Phaser.GameObjects.Extern {
    constructor(scene, size, fragmentSource, agentCount = 1000) {
        super(scene);
        this.scene = scene;
        this.game = scene.game;
        this.gl = this.game.renderer.gl;
        if (!(this.gl instanceof WebGL2RenderingContext)) {
            throw new Error('This simulation requires a WebGL2 context.');
        }
        if (!fragmentSource || !fragmentSource.includes('MODE_WAVE')) {
            throw new Error('Missing or invalid shaders/pathfinding.glsl');
        }
        this.size = size;
        this.agentCount = agentCount;
        this.agentTexSize = Math.ceil(Math.sqrt(agentCount));
        this.stepsPerFrame = 8;
        this.queryInterval = 64;
        this.maxSteps = Math.min(size * size, 65534); // 16-bit distance encoding
        this.steps = 0;
        this.finished = false;
        this.reachedLimit = false;
        this.queryPending = false;
        this.fragmentSource = fragmentSource;
        this.target = { x: Math.floor(size / 2), y: Math.floor(size / 2) };

        this.programs = {};
        this.textures = [];
        this.framebuffers = [];
        this.query = null;
        this.vao = null;
        this.lost = false;
        this.disposed = false;

        // Only CPU canvas ever used. The GPU samples these ACTUAL canvas pixels.
        // CPU walkability is retained strictly for hit testing/initial spawning.
        this.mazeCanvas = this.generateMazeCanvas();
        const pixelData = this.mazeCanvas.getContext('2d', { willReadFrequently: true })
            .getImageData(0, 0, size, size).data;
        this.cpuWalkable = new Uint8Array(size * size);
        for (let i = 0; i < size * size; i++) {
            this.cpuWalkable[i] = pixelData[i * 4] > 128 ? 1 : 0;
        }
        this.initialAgents = this.makeAgentData();
        this.target = this.findValidGoal(this.target.x, this.target.y);
        // Defer ALL raw GL calls until Phaser invokes Extern.render().
        this.needsInitialize = true;
        this.pendingGoal = null;

        this.onLost = () => { this.lost = true; };
        this.onRestored = () => {
            this.lost = false;
            this.needsInitialize = true; // Recreate from within Extern.render().
        };
        this.game.canvas.addEventListener('webglcontextlost', this.onLost);
        this.game.canvas.addEventListener('webglcontextrestored', this.onRestored);
    }

    generateMazeCanvas() {
        const s = this.size;
        const canvas = document.createElement('canvas');
        canvas.width = s;
        canvas.height = s;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, s, s);
        ctx.fillStyle = '#000000';
        const spacing = 64;
        for (let x = 0; x < s; x += spacing) {
            for (let y = 0; y < s; y += spacing) {
                if (Math.random() > 0.4) {
                    if (Math.random() > 0.5) ctx.fillRect(x, y, spacing, 4);
                    else ctx.fillRect(x, y, 4, spacing);
                }
            }
        }
        // Explicit wall; unlike strokeRect it is always exactly four pixels wide.
        ctx.fillRect(0, 0, s, 4);
        ctx.fillRect(0, s - 4, s, 4);
        ctx.fillRect(0, 0, 4, s);
        ctx.fillRect(s - 4, 0, 4, s);
        return canvas;
    }

    isPassable(x, y) {
        return x >= 0 && y >= 0 && x < this.size && y < this.size
            && this.cpuWalkable[y * this.size + x] === 1;
    }

    findValidGoal(x, y) {
        x = Math.max(0, Math.min(this.size - 1, Math.floor(x)));
        y = Math.max(0, Math.min(this.size - 1, Math.floor(y)));
        if (this.isPassable(x, y)) return { x, y };
        for (let radius = 1; radius < this.size; radius++) {
            for (let dx = -radius; dx <= radius; dx++) {
                for (const dy of [-radius, radius]) {
                    if (this.isPassable(x + dx, y + dy)) return { x: x + dx, y: y + dy };
                }
            }
            for (let dy = -radius + 1; dy < radius; dy++) {
                for (const dx of [-radius, radius]) {
                    if (this.isPassable(x + dx, y + dy)) return { x: x + dx, y: y + dy };
                }
            }
        }
        throw new Error('No traversable pixel exists in the maze.');
    }

    makeAgentData() {
        const width = this.agentTexSize;
        const data = new Uint8Array(width * width * 4);

//const columns = 32;
//const spacing = 30;
//for (let i = 0; i < 1000; i++) {
//    const x = i % columns;
//    const y = Math.floor(i / columns);
//    data[i * 4] = 32 + x * spacing;
//    data[i * 4 + 1] = 32 + y * spacing;
//    data[i * 4 + 2] = 0;
//    data[i * 4 + 3] = 1;
//}

        for (let i = 0; i < this.agentCount; ++i) {
            let x, y;
            do {
                x = 12 + Math.floor(Math.random() * (this.size - 12));
                y = 12 + Math.floor(Math.random() * (this.size - 12));
            } while (!this.isPassable(x, y));
            const k = i * 4;
            data[k] = (x >>> 8) & 255;
            data[k + 1] = x & 255;
            data[k + 2] = (y >>> 8) & 255;
            data[k + 3] = y & 255;
        }
        //console.log('Agent data:', data);
        return data;
    }

    compileProgram(vertexSource, fragmentSource) {
        const gl = this.gl;
        const compile = (type, source) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                const log = gl.getShaderInfoLog(shader);
                gl.deleteShader(shader);
                throw new Error('GLSL compile error: ' + log);
            }
            return shader;
        };
        const v = compile(gl.VERTEX_SHADER, vertexSource);
        const f = compile(gl.FRAGMENT_SHADER, fragmentSource);
        const program = gl.createProgram();
        gl.attachShader(program, v);
        gl.attachShader(program, f);
        gl.linkProgram(program);
        gl.deleteShader(v);
        gl.deleteShader(f);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const log = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error('GLSL link error: ' + log);
        }
        return program;
    }

    variantSource(mode) {
        // GLSL requires #version to be the first directive.
        return this.fragmentSource.replace('#version 300 es',
            '#version 300 es\n#define MODE_' + mode + ' 1');
    }

    // texImage2D(canvas) is the ONLY full-sized CPU->GPU image upload.
    createTexture(width, height, pixels = null, fromCanvas = false) {
        const gl = this.gl;
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // Agent textures contain packed coordinate bytes, not image colors.
        // Prevent alpha premultiplication from altering position data.
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        if (fromCanvas) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        } else {
            // 'null' means allocate VRAM without uploading a pixel array.
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0,
                          gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        }
        this.textures.push(texture);
        return texture;
    }

    createTarget(texture) {
        const gl = this.gl;
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                                gl.TEXTURE_2D, texture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            throw new Error('Incomplete GPU framebuffer (distance or agent texture).');
        }
        this.framebuffers.push(fbo);
        return fbo;
    }

    initializeGL() {
        const gl = this.gl;
        this.deleteGLResources();
        this.programs = {};
        for (const mode of ['WAVE', 'FRONTIER', 'AGENTS', 'DISPLAY']) {
            this.programs[mode] = this.compileProgram(QUAD_VERTEX, this.variantSource(mode));
        }
        this.programs.POINTS = this.compileProgram(POINT_VERTEX, POINT_FRAGMENT);
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);
        gl.disable(gl.DITHER);
        gl.disable(gl.BLEND);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.STENCIL_TEST);
        gl.disable(gl.SCISSOR_TEST);

        // Exactly one full maze upload; never update this texture afterward.
        this.maze = this.createTexture(this.size, this.size, this.mazeCanvas, true);
        this.dist = [
            this.createTexture(this.size, this.size),
            this.createTexture(this.size, this.size)
        ];
        this.distFbo = [this.createTarget(this.dist[0]), this.createTarget(this.dist[1])];
        // Separate, tiny initial agent-position upload (e.g. 32x32x4 bytes).
        this.agents = [
            this.createTexture(this.agentTexSize, this.agentTexSize, this.initialAgents),
            this.createTexture(this.agentTexSize, this.agentTexSize)
        ];
        this.agentFbo = [this.createTarget(this.agents[0]), this.createTarget(this.agents[1])];
        this.query = gl.createQuery();
        this.agentIndex = 0;
        this.resetGoal(this.target.x, this.target.y);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindVertexArray(null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        console.info('GLSL pathfinding initialized: one maze upload; agent positions GPU-resident.');
    }

    setGoal(x, y) {
        if (!this.isPassable(x, y)) return false;
        this.pendingGoal = { x: Math.floor(x), y: Math.floor(y) };
        return true;
    }

    resetGoal(x, y) {
        x = Math.floor(x); y = Math.floor(y);
        if (!this.isPassable(x, y)) return false;
        const gl = this.gl;
        this.target = { x, y };
        this.steps = 0;
        this.finished = false;
        this.reachedLimit = false;
        // Query from the previous goal must finish before it can be reused.
        if (this.queryPending) {
            gl.deleteQuery(this.query);
            this.query = gl.createQuery();
            this.queryPending = false;
        }
        this.distIndex = 0;
        gl.disable(gl.SCISSOR_TEST);
        gl.colorMask(true, true, true, true);
        for (const fbo of this.distFbo) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.viewport(0, 0, this.size, this.size);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return true;
    }

    bindTexture(unit, texture, program, uniformName) {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        const loc = gl.getUniformLocation(program, uniformName);
        if (loc !== null) gl.uniform1i(loc, unit);
    }

    usePass(name) {
        const gl = this.gl;
        const p = this.programs[name];
        gl.useProgram(p);
        const sizeLoc = gl.getUniformLocation(p, 'uSize');
        if (sizeLoc !== null && name !== 'POINTS') gl.uniform2i(sizeLoc, this.size, this.size);
        const goalLoc = gl.getUniformLocation(p, 'uGoal');
        if (goalLoc !== null) gl.uniform2i(goalLoc, this.target.x, this.target.y);
        if (name !== 'POINTS') {
            this.bindTexture(0, this.maze, p, 'uMaze');
            this.bindTexture(1, this.dist[this.distIndex], p, 'uDistance');
        }
        return p;
    }

    fullscreen() {
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
    }

    stepWavefront() {
        const gl = this.gl;
        const to = 1 - this.distIndex;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.distFbo[to]);
        gl.viewport(0, 0, this.size, this.size);
        this.usePass('WAVE');
        this.fullscreen();
        this.distIndex = to;
        this.steps++;
        if (this.steps >= this.maxSteps) {
            this.finished = true;
            this.reachedLimit = true;
        }
    }

    pollCompletion() {
        const gl = this.gl;
        if (this.queryPending && gl.getQueryParameter(this.query, gl.QUERY_RESULT_AVAILABLE)) {
            const more = gl.getQueryParameter(this.query, gl.QUERY_RESULT);
            this.queryPending = false;
            if (!more) this.finished = true;
        }
    }

    checkCompletion() {
        if (this.queryPending || this.finished) return;
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.game.canvas.width, this.game.canvas.height);
        // Query shader discards all fragments unless an unreached pixel borders
        // the distance field. Disable writes so the screen is unaffected.
        gl.colorMask(false, false, false, false);
        this.usePass('FRONTIER');
        gl.beginQuery(gl.ANY_SAMPLES_PASSED, this.query);
        this.fullscreen();
        gl.endQuery(gl.ANY_SAMPLES_PASSED);
        gl.colorMask(true, true, true, true);
        this.queryPending = true;
    }

    stepAgents() {
        const gl = this.gl;
        const to = 1 - this.agentIndex;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.agentFbo[to]);
        gl.viewport(0, 0, this.agentTexSize, this.agentTexSize);
        const p = this.usePass('AGENTS');
        this.bindTexture(2, this.agents[this.agentIndex], p, 'uAgents');
        gl.uniform2i(gl.getUniformLocation(p, 'uAgentTexSize'),
                     this.agentTexSize, this.agentTexSize);
        this.fullscreen();
        this.agentIndex = to;
    }

    drawDisplay(time) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        const width = this.game.canvas.width;
        const height = this.game.canvas.height;
        gl.viewport(0, 0, width, height);
        const p = this.usePass('DISPLAY');
        gl.uniform2f(gl.getUniformLocation(p, 'uViewport'), width, height);
        gl.uniform1f(gl.getUniformLocation(p, 'uTime'), time);
        this.fullscreen();
    }

    drawAgents() {
        const gl = this.gl;
        const p = this.usePass('POINTS');
        this.bindTexture(2, this.agents[this.agentIndex], p, 'uAgents');
        gl.uniform2i(gl.getUniformLocation(p, 'uAgentTexSize'),
                     this.agentTexSize, this.agentTexSize);
        gl.uniform2f(gl.getUniformLocation(p, 'uSize'), this.size, this.size);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArrays(gl.POINTS, 0, this.agentCount);
        gl.disable(gl.BLEND);
    }

    // Phaser 4 calls Extern.render within its own render loop. This is the only
    // location where raw gl state changes occur during normal gameplay.
    render() {
        if (this.lost || this.disposed) return;
        const gl = this.gl;
        if (this.needsInitialize) {
            this.initializeGL();
            this.needsInitialize = false;
        }
        gl.bindVertexArray(this.vao);
        gl.disable(gl.SCISSOR_TEST);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.STENCIL_TEST);
        gl.disable(gl.BLEND);
        gl.disable(gl.DITHER); // RGBA8 integer-encoded distances must stay exact
        gl.colorMask(true, true, true, true);
        if (this.pendingGoal) {
            this.resetGoal(this.pendingGoal.x, this.pendingGoal.y);
            this.pendingGoal = null;
        }
        this.pollCompletion();
        for (let i = 0; i < this.stepsPerFrame && !this.finished; i++) {
            this.stepWavefront();
        }
        if (!this.finished && this.steps % this.queryInterval === 0) {
            this.checkCompletion();
        }
        this.stepAgents();
        this.drawDisplay(performance.now());
        this.drawAgents();
        // Phaser's Extern render node restores its tracked WebGL state on exit.
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindVertexArray(null);
    }

    deleteGLResources() {
        const gl = this.gl;
        if (!gl || this.lost) return;
        for (const p of Object.values(this.programs || {})) gl.deleteProgram(p);
        for (const fbo of this.framebuffers || []) gl.deleteFramebuffer(fbo);
        for (const texture of this.textures || []) gl.deleteTexture(texture);
        if (this.vao) gl.deleteVertexArray(this.vao);
        if (this.query) gl.deleteQuery(this.query);
        this.programs = {};
        this.textures = [];
        this.framebuffers = [];
        this.vao = null;
        this.query = null;
        this.queryPending = false;
    }

    destroy(fromScene) {
        if (this.disposed) return;
        this.disposed = true;
        this.game.canvas.removeEventListener('webglcontextlost', this.onLost);
        this.game.canvas.removeEventListener('webglcontextrestored', this.onRestored);
        this.deleteGLResources();
        super.destroy(fromScene);
    }
}
