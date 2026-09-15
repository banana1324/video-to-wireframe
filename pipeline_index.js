const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

// build_wireframe.py creates these names from the object name.
const vertices = bananaVertices;
const edges = bananaEdges;

const BACKGROUND = "#202020";
const FOREGROUND = "#50ff90";
let spin = 0;
let lastTime = 0;

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
    const delta = lastTime ? (time - lastTime) / 1000 : 0;
    lastTime = time;
    spin += delta * 0.8;

    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = FOREGROUND;
    ctx.lineWidth = 1;
    ctx.beginPath();

    const projected = vertices.map((vertex) => {
        const rotated = rotateY(vertex, spin);
        const z = rotated.z + 3;
        return {
            x: (rotated.x / z + 1) * canvas.width / 2,
            y: (1 - rotated.y / z) * canvas.height / 2,
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
