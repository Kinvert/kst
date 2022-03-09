
console.clear();
var canv = document.getElementById("myCanvas");
canv.setAttribute('tabindex', '0')
canv.focus()
var ctx = canv.getContext("2d");
var canv2 = document.getElementById("myCanvas2");
canv2.setAttribute('tabindex', '1')
canv2.focus()
var ctx2 = canv2.getContext("2d");
document.addEventListener("keydown", keypress);
document.addEventListener("keyup", keyrelease);

var track = new Image();
var track2 = new Image();
var circuit = 1;
var fname = 'img/circuits/circuit-' + String(circuit) + '.jpg';
track.src = fname;
track2.src = fname;
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
var drag = 0.15;

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
const farwidth = far * Math.sin(fov);

function gameloop() {
    if (v > maxv) {
        v = maxv;
    }
    if (ang > 2*Math.PI) {
        ang = 0;
    }
    if (throttle == 0 && v > 0) {
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
    var flx = px + Math.cos(ang - fovh) * far; // Far Left X
    var fly = py + Math.sin(ang - fovh) * far;
    ctx.beginPath();
    ctx.arc(flx, fly, 5, 0, 2*Math.PI, false);
    ctx.fillStyle = 'white';
    ctx.fill();
    var frx = px + Math.cos(ang + fovh) * far; // Far Right X
    var fry = py + Math.sin(ang + fovh) * far;
    ctx.beginPath();
    ctx.arc(frx, fry, 5, 0, 2*Math.PI, false);
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




    ctx2.save()
    ctx2.rotate(-Math.PI/2-ang);
    ctx2.drawImage(canv, -flx, -fly); // top left
    //ctx2.drawImage(canv, -flx-h+far, -fly+w/2-farwidth/2);
    //ctx2.transform(2, Math.PI/4, Math.PI/4, 0, -flx, -fly);
    ctx2.restore();
    /*
    ctx2.drawImage(canv, 0, 0);
    var imageData = ctx2.createImageData(w, h/2);
    for (var y = 1; y < h/2+1; y++) {
        var depth = 1.0*y / h;
        var startx = (flx - nlx) / depth + nlx;
        var starty = (fly - nly) / depth + nly;
        var endx   = (frx - nrx) / depth + nrx;
        var endy   = (fry - nry) / depth + nry;

        for (var x = 0; x < w; x++) {
            var samplewidth = 1.0*x / w;
            var samplex = (endx - startx) * samplewidth + startx;
            var sampley = (endy - starty) * samplewidth + starty;
            var color = ctx.getImageData(samplex, sampley, 1, 1);
            //imageData.data[(w*y + x) * 4 + 0] = y/h * 255;
            //imageData.data[(w*y + x) * 4 + 1] = y/h * 255;
            //imageData.data[(w*y + x) * 4 + 2] = x/w * 255;
            //imageData.data[(w*y + x) * 4 + 3] = 255;
            imageData.data[(w*y + x) * 4 + 0] = color[0];
            imageData.data[(w*y + x) * 4 + 1] = color[1];
            imageData.data[(w*y + x) * 4 + 2] = color[2];
            imageData.data[(w*y + x) * 4 + 3] = 255;
        }
        ctx2.putImageData(imageData, 0, h/2);
    }
    */
    
}

function keypress(evt) {
    switch(evt.keyCode) {
        case 37: // Left
        case 65: // a
            ang -= Math.PI / turn_pi_frac;
            break;
        case 38: // Up
        case 87: // w
            v += accel;
            throttle = 1
            break;
        case 39: // Right
        case 68: // d
            ang += Math.PI / turn_pi_frac;
            break;
        case 40: // Down
        case 83: // s
            v -= brake;
            break;
    }
}

function keyrelease(evt) {
    switch(evt.keyCode) {
        case 38:
        case 87:
            throttle = 0;
    }
}