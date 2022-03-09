
console.clear();
var canv = document.getElementById("myCanvas");
canv.setAttribute('tabindex', '0')
canv.focus()
var ctx = canv.getContext("2d");
document.addEventListener("keydown", keypress);
document.addEventListener("keyup", keyrelease);

var track = new Image();
var circuit = 1;
var fname = 'img/circuits/circuit-' + String(circuit) + '.jpg';
track.src = fname;
track.onload = function(){
    draw();
};

const w = 640; // canv.style.width;
const h = 480; // canv.style.height;

/*
PHYSICS CONSTANTS
*/
var fps = 15; // Frames Per Second
setInterval(gameloop, 1000/fps);
var turn_pi_frac = 15; // ang += Math.PI / turn_pi_frac
var accel = 0.5;
var brake = 2.0;
var maxv = 5.0;
var drag = 0.3;

if (circuit == 1) {
    var px = 315;
    var py = 380;
    var ang = 0;
}
else {
    var px = w/2; // Position X
    var py = h/2;
    var ang = 0;
}
var vx = 0; // Velocity X
var vy = 0;
var v = 0;
var vang = 0;
var throttle = 0;

const near = 10; // Close end frustrum
const far  = 75; // Far end frustrum
const fov  = 3.14159 / 3.0; // Field of view
const fovh = fov / 2.0; // Half the field of view

function gameloop() {
    if (v > maxv) {
        v = maxv;
    }
    if (ang > 2*Math.PI) {
        ang = 0;
    }
    if (throttle == 0 && v > 0) {
        console.log('slowing down');
        v -= drag;
    }
    if (v < 0) {
        v = 0;
    }
    vx = v * Math.cos(ang);
    vy = v * Math.sin(ang);
    px = px + vx;
    py = py + vy;
    //ang = ang + vang;
    draw();
}

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
            console.log("left"); // Turn Left
            ang -= Math.PI / turn_pi_frac;
            break;
        case 38:
            console.log("up"); // Accelerate
            v += accel;
            throttle = 1
            break;
        case 39:
            console.log("right"); // Turn Right
            ang += Math.PI / turn_pi_frac;
            break;
        case 40:
            console.log("down"); // Brakes
            v -= brake;
            break;
    }
    
}

function keyrelease(evt) {
    switch(evt.keyCode) {
        case 38:
            throttle = 0;
            console.log('THROTTLE RELEASED');
    }
}