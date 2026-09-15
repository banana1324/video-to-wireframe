const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

if (typeof modelVertices === "undefined" || typeof modelEdges === "undefined") {
    ctx.fillStyle = "#ffffff";
    ctx.font = "20px sans-serif";
    ctx.fillText("Missing model-data.js. Run build_surface.py first.", 20, 40);
    throw new Error("Load model-data.js before index.js.");
}

const vertices = modelVertices;
const edges = modelEdges;

const BACKGROUND = "#202020";
const FOREGROUND = "#50ff90";

let spin = 0;
let lastTime = 0;
let zoom = 1;

canvas.addEventListener("wheel", (event) => {
    event.preventDefault();

    // Handle scrolling measured in pixels, lines, or pages.
    const unit = event.deltaMode === 1 ? 16
        : event.deltaMode === 2 ? canvas.clientHeight : 1;

    const delta = event.deltaY * unit;

    // Scroll up increases zoom; scroll down decreases it.
    zoom *= Math.exp(-delta * 0.0015);
    zoom = Math.max(0.2, Math.min(6, zoom));
}, { passive: false });

function rotateY(point, angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);

    return {
        x: point.x * c + point.z * s,
        y: point.y,
        z: -point.x * s + point.z * c,
    };
}

function draw(time) {
    const delta = lastTime
        ? Math.min((time - lastTime) / 1000, 0.05)
        : 0;

    lastTime = time;
    spin += delta * 0.8;

    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = FOREGROUND;
    ctx.lineWidth = 1;
    ctx.beginPath();

    const scale = Math.min(canvas.width, canvas.height) / 2 * zoom;

    const projected = vertices.map((vertex) => {
        const rotated = rotateY(vertex, spin);
        const distance = rotated.z + 3;

        if (distance <= 0.01) return null;

        return {
            x: canvas.width / 2 + rotated.x / distance * scale,
            y: canvas.height / 2 - rotated.y / distance * scale,
        };
    });

    for (const [a, b] of edges) {
        const start = projected[a];
        const end = projected[b];

        if (!start || !end) continue;

        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
    }

    ctx.stroke();
    requestAnimationFrame(draw);
}

requestAnimationFrame(draw);