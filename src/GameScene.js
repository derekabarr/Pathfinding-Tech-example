import { GpuPathfindingVisualizer } from './GpuPathfindingVisualizer.js';

export class GameScene extends Phaser.Scene {
    constructor() {
        super('GameScene');
    }

    preload() {
        // Phaser 4 caches GLSL as plain text, not a Phaser 3 pipeline shader.
        this.load.text('pathShader', 'shaders/pathfinding.glsl');
    }

    create() {
        try {
            const shader = this.cache.text.get('pathShader');
            this.visualizer = new GpuPathfindingVisualizer(this, 1024, shader, 1000);
            this.add.existing(this.visualizer);

            // Hit testing reads the original CPU canvas pixel mask, not GPU data.
            // There are NO GPU readbacks and no new maze uploads on mouse clicks.
            this.input.on('pointerdown', (pointer) => {
                const bounds = this.game.canvas.getBoundingClientRect();
                const x = Math.floor((pointer.event.clientX - bounds.left) / bounds.width * 1024);
                const y = Math.floor((pointer.event.clientY - bounds.top) / bounds.height * 1024);
                if (!this.visualizer.setGoal(x, y)) {
                    this.showMessage('Click an open (non-black) area to set the destination.');
                    return;
                }
                this.showMessage(`Goal: (${x}, ${y}) — computing on GPU`);
            });

            this.input.keyboard?.on('keydown-OPEN_BRACKET', () => {
                this.visualizer.stepsPerFrame = Math.max(1, this.visualizer.stepsPerFrame - 1);
            });
            this.input.keyboard?.on('keydown-CLOSED_BRACKET', () => {
                this.visualizer.stepsPerFrame = Math.min(64, this.visualizer.stepsPerFrame + 1);
            });
            this.hud = document.getElementById('stats');
            this.message = document.getElementById('message');
            this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => {
                this.visualizer?.destroy();
            });
        } catch (err) {
            console.error(err);
            document.getElementById('message').textContent = err.message;
            throw err;
        }
    }

    showMessage(message) {
        if (this.message) this.message.textContent = message;
    }

    update(time) {
        if (!this.visualizer || !this.hud || time - (this.lastHudUpdate || 0) < 200) return;
        this.lastHudUpdate = time;
        const v = this.visualizer;
        const state = v.reachedLimit ? 'MAX DISTANCE / PARTIAL' : v.finished ? 'COMPLETE' : 'PROPAGATING';
        this.hud.textContent = `${v.agentCount} GPU agents  ·  ${v.steps} wave passes  ·  ` +
            `${v.stepsPerFrame} passes/frame  ·  ${state}`;
    }
}
