/*
TODO
    Grayscale noise to create hills
    Better comments to document code
    How to handle edge of map - probably just fill green
    Improve physics
    Sky
    Background, like far away hills etc
    Reduce distortion, straight lines right next to the car seem to curve
*/

// https://www.youtube.com/watch?v=PyQNfsGUnQA
class ReplayMemory {
    constructor(capacity) {
        this.capacity = capacity;
        this.memory = [];
        this.push_count = 0;
    }

    push(experience) {
        if(this.memory.length < this.capacity){
            this.memory.unshift(experience);
        }
        else {
            this.memory.pop();
            this.memory.unshift(experience);
        }
        this.push_count += 1;
    }

    // https://stackoverflow.com/questions/11935175/sampling-a-random-subset-from-an-array
    get_rand_memory(arr, size) {
        var shuffled = arr.slice(0), i = arr.length, min = i - size, temp, index;
        while (i-- > min) {
            index = Math.floor((i + 1) * Math.random());
            temp = shuffled[index];
            shuffled[index] = shuffled[i];
            shuffled[i] = temp;
        }
        return shuffled.slice(min);
    }

    can_provide_sample(batch_size) {
        return self.memory.length >= batch_size;
    }
}

// //https://www.youtube.com/watch?v=PyQNfsGUnQA
class Experience {
    constructor() {
        this.state = 'state';
        this.action = 'action';
        this.next_state = 'next_state';
        this.reward = 'reward';
    }
}

// //https://www.youtube.com/watch?v=PyQNfsGUnQA
class EpsGreedyStrat {
    constructor(start, end, decay) {
        this.start = start;
        this.end = end;
        this.decay = decay;
    }

    get_exploration_rate(current_step) {
        var expl_val = this.start - current_step * this.decay;
        if (expl_val < this.end) {
            expl_val = this.end;
        }
        return expl_val;
    }
}

// //https://www.youtube.com/watch?v=PyQNfsGUnQA
class Agent {
    constructor(strategy, num_actions) {
        this.current_step = 0;
        this.strategy = strategy;
        this.num_actions = num_actions
    }

    select_action(state, policy_net) {
        // policy net is the name of the deep neural net
        var rate = this.strategy.get_exploration_rate(this.current_step);
        this.current_step += 1;
        if (rate > Math.random()) { // Explore
            return Math.floor(3*Math.random());
        }
        else { // Exploit
            return policy_net.predict(state).argMax(-1).dataSync()[0];
        }
    }
}

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
var circuit = 1;
var fname = 'img/circuits/circuit-' + String(circuit) + '.jpg';
track.src = fname;
track.onload = function(){
    draw();
};

const W = 640; // canv.style.width;
const H = 480; // canv.style.height;
var horiz = H/2; // Horizon y height in canvas
var log_vars = 0;

/*
PHYSICS CONSTANTS==================================================================
*/
const ML = true;
var fps = 15; // Frames Per Second
var turn_pi_frac = 20; // ang += Math.PI / turn_pi_frac
var accel = 0.5;       // Acceleration
var brake = 2.0;       // Braking
var maxv = 5.0;        // Max Velocity
var drag = 0.15;       // Drag

if (circuit == 1) {
    var px = 330; // Initial Position X
    var py = 376; // Initial Position Y
    var ang = 0;  // Initial Angle (Radians)
}
else {
    var px = W/2; // Initial Position X
    var py = H/2; // Initial Position Y
    var ang = 0;  // Initial Angle (Radians)
}
var vx = 0;       // Initial Velocity X
var vy = 0;       // Initial Velocity Y
if (ML) {
    var v = maxv;     // Initial Velocity Magnitude
}
else {
    var v = 0;        // Initial Velocity Magnitude
}
var vang = 0;     // Initial Velocity Angle
var throttle = 0; // Initial Throttle Boolean False

const near = 10;            // Close end frustrum
const far  = 75;            // Far end frustrum
const fov  = 3.14159 / 3.0; // Field of view
const fovh = fov / 2.0;     // Half the field of view
const farwidth = far * Math.sin(fov);
const height = 20;
ctx.drawImage(track, 0, 0);

/*
ML ==========================================================================
*/
var batch_size = 256;
var lr = 0.001;
var expl_rate_start = 1;
var expl_rate_end = 0.01;
var expl_rate_decay = 0.01;
var episodes = 2;
var max_timesteps = 100;
var num_actions = 3;
var memory_size = 10;
var strategy = new EpsGreedyStrat(expl_rate_start, expl_rate_end, expl_rate_decay);
var agent = new Agent(strategy, num_actions);
var memory = new ReplayMemory(memory_size);

function game_reset() {
    if (circuit == 1) {
        px = 330; // Initial Position X
        py = 376; // Initial Position Y
        ang = 0;  // Initial Angle (Radians)
    }
    else {
        px = W/2; // Initial Position X
        py = H/2; // Initial Position Y
        ang = 0;  // Initial Angle (Radians)
    }
    vx = 0;       // Initial Velocity X
    vy = 0;       // Initial Velocity Y
    if (ML) {
        v = maxv;     // Initial Velocity Magnitude
    }
    else {
        v = 0;        // Initial Velocity Magnitude
    }
    var vang = 0;     // Initial Velocity Angle
    var throttle = 0; // Initial Throttle Boolean False
}

function do_the_thing(action) {
    if (action == 0) { // Turn Left
        ang -= Math.PI / turn_pi_frac;
    }
    else if (action == 2) { // Turn Right
        ang += Math.PI / turn_pi_frac;
    }
}

function get_state() {
    var imgForTensor = ctx2.getImageData(0, horiz, W, H-horiz)
    var tensor = tf.browser.fromPixels(imgForTensor);
    return tf.reshape(tensor, [-1, horiz, W, 3]);
}

function carrot_stick(action) {
    var color = ctx.getImageData(px, py, 1, 1);
    var red = color.data[0];
    var green = color.data[1];
    var blue = color.data[2];
    //console.log(red, green, blue);
    var score = 0;
    var endrun = 0;
    if (red == green == blue && red > 2) {
        score += 100; // WHITE lap
    }
    else if (red = 69 && green == 142 && blue > 49 && blue < 55) {
        score -= 100; // GREEN crash
        endrun = 1;
    }
    else if (red > 225 && green > 230 && blue > 35 && blue < 50) {
        score += 10; // YELLOW waypoint
    }
    else if (red > 230 && green < 20 && blue < 20) {
        score += 5; // RED hitting racing line
    }
    if ( action == 0 && action == 2) {
        score -= 1; // Penalize many turns
    }
    
    return {score: score, endrun: endrun};
}

window.onload = function(){
    console.log('WINDOW ON LOAD');
    const policy_net = tf.sequential();
    policy_net.add(tf.layers.conv2d({inputShape: [H/2, W, 3],
                                    name: 'conv2d_1',
                                    filters: 7,
                                    kernelSize: 7,
                                    padding: 'same',
                                    activation: 'relu'}));
    policy_net.add(tf.layers.flatten());
    policy_net.add(tf.layers.dense({name: 'dense_1',
                                    units: 128,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'relu'}));
    policy_net.add(tf.layers.dense({name: 'dense_2',
                                    units: 3,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'softmax'}));
    var target_net = policy_net; // This isn't for training this is for evaluating
    if (ML == false) {
        setInterval(gameloop, 1000/fps);
    }
    else {
        // BIG ML LOOP
        var timesteps_history = [];
        var px_history = [];
        var py_history = [];
        for (var eps = 0; eps < episodes; eps += 1) {
            console.log(' ');
            console.log(' ');
            console.log('                NEW GAME              ');
            game_reset()
            var endrun = 0;
            var reward = 0;
            var state = get_state();
            for (var timestep=0; timestep < max_timesteps; timestep++) {
                var action = agent.select_action(state, policy_net);
                do_the_thing(action); // Take action
                gameloop(); // Run the timestep through super physics
                let {reward, endrun} = carrot_stick(action);
                draw_car(px, py, ang);
                px_history.push(px);
                py_history.push(py);
                if (endrun == 1) {
                    console.log('            GAME OVER MAN GAME OVER');
                    timesteps_history.push(timestep);
                    draw_drive(px_history, py_history);
                    break;
                }
                console.log('timestep = ', timestep);
                var reward_tensor = tf.tensor([reward]);
                // train
            }
        }
    }
    console.log('===========OVER==============');
    console.log('timesteps_history = ', timesteps_history);
};

function gameloop() {
    if (v > maxv) { // Don't exceed max speed
        v = maxv;
    }
    if (ang > 2*Math.PI) { // Avoid very large angles
        ang = 0;
    }
    if (throttle == 0 && v > 0) { // No throttle so coast to a stop
        v -= drag;
    }
    if (v < 0) { // No reverse currently
        v = 0;
    }
    if (ML) {
        v = maxv;
    }
    vx = v * Math.cos(ang); // Update velocities and positions
    vy = v * Math.sin(ang);
    px = px + vx;
    py = py + vy;
    draw();
    if (ML == false) { // Draw car now since it wont hurt carrot_stick
        draw_car(px, py, ang);
    }

    // Check State

    // Kill

    // Give Rewards

    // Update weights idk
}

function draw_car(px, py, ang) {
    // Placeholder Car
    ctx.strokeStyle = 'magenta';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(px-10*Math.cos(ang), py-10*Math.sin(ang));
    ctx.lineTo(px+10*Math.cos(ang), py+10*Math.sin(ang));
    ctx.stroke();
}

function draw_drive(px_history, py_history) {
    ctx.strokeStyle = 'magenta';
    ctx.lineWidth = 2;
    ctx.moveTo(px_history[0], py_history[0]);
    for (var i = 1; i < px_history.length; i++) {
        if (Math.abs(px_history[i]-px_history[i-1]) > 10 || Math.abs(py_history[i]-py_history[i-1]) > 10) {
            ctx.stroke();
            ctx.moveTo(px_history[i], py_history[i]);
        }
        ctx.lineTo(px_history[i], py_history[i]);
    }
    ctx.stroke();
}

function draw() {
    
    ctx.drawImage(track, 0, 0);
    //draw_fov();

    

    /* TINY TOP VIEW */
    /*
    ctx2.save()
    ctx2.rotate(-Math.PI/2-ang);
    ctx2.drawImage(canv, -flx, -fly);
    ctx2.restore();
    */
    
    var cos_ang = Math.cos(ang);
    var sin_ang = Math.sin(ang);

    var perspective = ctx2.createImageData(W, H);
    var topDown = ctx.getImageData(0, 0, W, H);
    // Mode 7 like algorithm
    // http://coranac.com/tonc/text/mode7.htm
    for (var i = 4*W*(horiz); i < topDown.data.length; i += 4) {
        var imgy = Math.floor(i/(4*W)); // Image Plane Y Coordinate. 4 bytes per pixel. w pixels wide. Increment Y
        var imgx = Math.floor((i/4)%W)-(W/2); // Imag Plane X Coord. 4 bytes/pix. w pix wide. 0 at center not at left
        var sf = imgy/height;                  // sf is a scaling factor making near things large
        var z = Math.floor(5000/(imgy-horiz)); // Depth Map
        var z2 = Math.floor( 0.75*(W/2)*height / (imgy-horiz) ); // Depth Map attempt 2
        var view_angley = imgy-(horiz);         // y view angle this pass
        var view_anglex = view_angley*(W/H); // x view angle this pass

        var xval = (imgx/(sf*view_anglex))*(W/4);
        var yval = (height/view_angley)*(horiz);

        var xval2 = imgx/(z);
        var yval2 = imgy/z;

        // z2 and yval are nearly identical
        var xprime = Math.floor(px - (xval * sin_ang) + (z2 * cos_ang)); // top down x coord of pixel
        var yprime = Math.floor(py + (xval * cos_ang) + (z2 * sin_ang)); // top down y coord of pixel

        if (i % 100000 == 0 && log_vars == 1) {
            console.log(' ');
            console.log(' ');
            console.log('i', i, 'z', z, 'z2', z2);
            console.log('ang', ang, 'cos_ang', cos_ang, 'sin_ang', sin_ang);
            console.log('imgx', imgx, 'imgy', imgy, 'sf', sf, 'viewx', view_anglex, 'viewy', view_angley);
            console.log('xval2', xval2, 'yval2', yval2);
            console.log('xval', xval, 'yval', yval, 'xp', xprime, ' yp', yprime);
        }
        
        if(xprime >= 0 && xprime <= W && yprime >= 0 && yprime <= H){
            var idx = ((yprime * W) + xprime) * 4; // Get pixel index based on pixel (x,y)
            perspective.data[i] = topDown.data[idx]; // Get color from original top view image
            perspective.data[i+1] = topDown.data[idx+1];
            perspective.data[i+2] = topDown.data[idx+2];
            perspective.data[i+3] = topDown.data[idx+3];

            // TOO SLOW
            //var color = ctx.getImageData(xprime, yprime, 1, 1);
            //perspective.data[i] = color.data[0];
            //perspective.data[i+1] = color.data[1];
            //perspective.data[i+2] = color.data[2];
            //perspective.data[i+3] = color.data[3];
        }
    }
    log_vars = 0;
    ctx2.putImageData(perspective, 0, 0);
    // Placeholder Sky
    //ctx2.beginPath()
    //ctx2.rect(0, 0, W, h/2+8);
    //ctx2.fillStyle = '#4488FF';
    //ctx2.fill()

    // TODO Difference between this img and last img to only show changed pixels
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
        case 80: // p
            log_vars = 1; // Console.log if log_vals == 1
            break;
    }
}

function keyrelease(evt) {
    switch(evt.keyCode) {
        case 38:
        case 87:
            throttle = 0; // Coasting
    }
}

function draw_fov() {
    // Draw Field of View
    var flx = px + Math.cos(ang - fovh) * far; // Far Left X
    var fly = py + Math.sin(ang - fovh) * far;
    ctx.beginPath();
    ctx.arc(flx, fly, 5, 0, 2*Math.PI, false);
    ctx.fillStyle = 'magenta';
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
}