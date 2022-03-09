
console.clear();
var canv = document.getElementById("myCanvas");
canv.setAttribute('tabindex', '0')
canv.focus()
var ctx = canv.getContext("2d");
document.addEventListener("keydown", keypress);

var track = new Image();
track.src = 'img/circuits/circuit-1.jpg';
track.onload = function(){
    draw();
};

const w = 640; // canv.style.width;
const h = 480; // canv.style.height;

var px = w/2; // Position X
var py = h/2;
var ang = 0; // Angle
var vx = 0; // Velocity X
var vy = 0;
var vang = 0;

const near = 10; // Close end frustrum
const far  = 30; // Far end frustrum
const fov  = 3.14159 / 3.0; // Field of view
const fovh = fov / 2.0; // Half the field of view

function draw() {
    ctx.drawImage(track, 0, 0);
    var frx = px + Math.cos(ang - fovh) * far; // Far Right X
    var fry = py + Math.sin(ang - fovh) * far;
    ctx.beginPath();
    ctx.arc(frx, fry, 5, 0, 2*Math.PI, false);
    ctx.fillStyle = 'white';
    ctx.fill();

    var flx = px + Math.cos(ang + fovh) * far; // Far Left X
    var fly = py + Math.sin(ang + fovh) * far;
    ctx.beginPath();
    ctx.arc(flx, fly, 5, 0, 2*Math.PI, false);
    ctx.fill();

    var nlx = px + Math.cos(ang + fovh) * near; // Near Left X
    var nly = py + Math.sin(ang + fovh) * near;
    ctx.beginPath();
    ctx.arc(nlx, nly, 5, 0, 2*Math.PI, false);
    ctx.fill();

    var nrx = px + Math.cos(ang - fovh) * near; // Near Right X
    var nry = py + Math.sin(ang - fovh) * near;
    ctx.beginPath();
    ctx.arc(nrx, nry, 5, 0, 2*Math.PI, false);
    ctx.fill();
}

function keypress(evt) {
    switch(evt.keyCode) {
        case 37:
            console.log("left");
            vx -= 1;
            break;
        case 38:
            console.log("up");
            vy -= 1;
            break;
        case 39:
            console.log("right");
            vx += 1;
            break;
        case 40:
            console.log("down");
            vy += 1;
            break;
    }
    px = px + vx;
    py = py + vy;
    ang = ang + vang;
    draw();
}