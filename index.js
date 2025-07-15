/*
TODO
    Get it to actually converge
    Multiple vehicles at one time
    Fix memory leak
    Grayscale noise to create hills
    Better comments to document code
    How to handle edge of map - probably just fill green
    Improve physics
    Sky
    Background, like far away hills etc
    Reduce distortion, straight lines right next to the car seem to curve
    https://deeplizard.com/learn/video/ewRw996uevN

    BELLMAN
        https://stackoverflow.com/questions/51230542/bellmans-equations-loss-in-tfjs
*/

const ML = false;
const ISOMETRIC = true;
const WHISKERS = false;
const whisker_angles = [-Math.PI/4, -Math.PI/6, 0,
                        Math.PI/6, Math.PI/4];
const max_whisker_length = 100; // 200
var batch_size = 64;
var lr = 0.1; // 0.01
var expl_rate_start = 1;
var expl_rate_end = 0.25;
var expl_rate_decay = 0.01;
var gamma = 0.99;
var discountRate = 0.99;
var episodes = 10;
var max_timesteps = 400;
var num_actions = 3;
var memory_size = 256; // replayBufferSize

// //https://www.youtube.com/watch?v=PyQNfsGUnQA
class Experience {
    constructor(state, action, next_state, reward, endrun) {
        this.state = state;
        this.action = action;
        this.next_state = next_state;
        this.reward = reward;
        this.endrun = endrun;
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

    select_action(state, policyNet) {
        // policy net is the name of the deep neural net
        var rate = this.strategy.get_exploration_rate(this.current_step);
        this.current_step += 1;
        if (rate > Math.random()) { // Explore
            return Math.floor(3*Math.random());
        }
        else { // Exploit
            // https://github.com/tensorflow/tfjs-examples/blob/master/snake-dqn/agent.js#L95
            return policyNet.predict(state).argMax(-1).dataSync()[0];
        }
    }

    reset_steps() {
        this.current_step = 0;
    }
}

class QValues {
    // https://youtu.be/ewRw996uevM?t=718
    constructor() {
        console.log('QValues');
    }
    
    //device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    //staticmethod
    //get_current(policyNet, states, actions) {
        //return policyNet(states).gather(dim=1, index=actions.unqueeze(-1));
    //}
    /*
    get_next(target_net, next_states) {
        final_state_locations = next_states.flatten(start_dim=1)
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == false)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
    }
    */
}

console.clear();
var canv = document.getElementById("myCanvas");
canv.setAttribute('tabindex', '0')
var ctx = canv.getContext("2d");
var canv2 = document.getElementById("myCanvas2");
canv2.setAttribute('tabindex', '1')
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

const W = 640;
const H = 480;
var horiz = H/2;
var log_vars = 0;

/*
PHYSICS CONSTANTS==================================================================
*/
var fps = 15;
var turn_pi_frac = 20; // ang += Math.PI / turn_pi_frac
var accel = 0.5;
var brake = 2.0;
if (ML == true) {
    var maxv = 5.0;
}
else {
    var maxv = 7.0;
}      
var drag = 0.15;

if (circuit == 1) {
    var px = 330;
    var py = 376;
    var ang = 0;
}
else {
    var px = W/2;
    var py = H/2;
    var ang = 0;
}
var vx = 0;
var vy = 0;
if (ML) {
    var v = maxv;
}
else {
    var v = 0;
}
var vang = 0;
var throttle = 0;

const near = 10;            // Close end frustrum
const far  = 75;            // Far end frustrum
const fov  = 3.14159 / 3.0; // Field of view
const fovh = fov / 2.0;     // Half the field of view
const farwidth = far * Math.sin(fov);
const height = 20;
ctx.drawImage(track, 0, 0);

function game_reset() {
    if (circuit == 1) {
        get_random_start();
    }
    else {
        px = W/2;
        py = H/2;
        ang = 0;
    }
    vx = 0;
    vy = 0;
    if (ML) {
        v = maxv;
    }
    else {
        v = 0;
    }
    vang = 0;
    throttle = 0;
}

function do_the_thing(action) {
    if (action == 0) { // Turn Left
        ang -= Math.PI / turn_pi_frac;
    }
    else if (action == 2) { // Turn Right
        ang += Math.PI / turn_pi_frac;
    }
}

function get_whisker_ends(px, py, ang) {
    var whisker_lengths = [];
    var whisker_ends = [];
    for (var a = 0; a < whisker_angles.length; ++a) {
        var angle = ang + whisker_angles[a];
        var hit_green = false;
        var l = 0; // Whisker Length
        const stepsize = 5;
        while (hit_green == false && l < max_whisker_length) {
            l += stepsize;
            var whisker_x = px + l * Math.cos(angle);
            var whisker_y = py + l * Math.sin(angle);
            var color = ctx.getImageData(Math.floor(whisker_x), Math.floor(whisker_y), 1, 1);
            var red = color.data[0];
            var green = color.data[1];
            var blue = color.data[2];
            if (red == 69 && green == 142 && blue > 49 && blue < 55) {
                hit_green = true;
                break;
            }
        }
        l = l / max_whisker_length;
        whisker_lengths.push(l);
        whisker_ends.push(Math.floor(whisker_x));
        whisker_ends.push(Math.floor(whisker_y));
    }
    return {whisker_lengths: whisker_lengths, whisker_ends: whisker_ends};
}

function make_state_tensor(state) {
    var tensor = tf.tensor2d([state]);
    return tensor;
}

function get_state(px, py, ang) {
    var {whisker_lengths} = get_whisker_ends(px, py, ang);
    return whisker_lengths;
}

function make_action_one_hot(action) {
    // action is an int of the index. This turns it to 1 hot [0 1 0 ]
    var action_one_hot = [];
    for (var aoh = 0; aoh < num_actions; aoh++) {
        if (aoh == action) {
            action_one_hot.push(parseInt(1));
        }
        else {
            action_one_hot.push(parseInt(0));
        }
    }
    return action_one_hot;
}

function carrot_stick(px, py, action) {
    var color = ctx.getImageData(px, py, 1, 1);
    var red = color.data[0];
    var green = color.data[1];
    var blue = color.data[2];
    var score = 0;
    if (red > 200 && green > 200 && blue > 200) {
        score += 1; // WHITE lap
    }
    else if (red > 60 && red < 75 && green > 100 && green < 175 && blue > 49 && blue < 55) {
        score -= 1; // GREEN crash
    }
    else if (red > 200 && green > 200 && blue < 50) {
        score += 0.5; // YELLOW waypoint
    }
    else if (red > 230 && green < 20 && blue < 20) {
        score += 0.05; // RED hitting racing line
    }
    if ( action == 0 || action == 2) {
        score -= 0.01; // Penalize many turns
    }
    score -= 0.001; // make them want to drive as fast as possible
    if (score > 1) {
        score = 1;
    }
    else if (score < -1) {
        score = -1;
    }
    if (score > 0) {
        console.log('score = ', score);
    }
    return score;
}

function get_random_start() {
    if (circuit == 1){
        var choices = [];
        // Xmin, Ymin, Xmax, Ymax, Angle
        choices.push([330, 360, 475, 395, 0]);
        choices.push([490, 350, 520, 370, -Math.PI/4.0]);
        choices.push([490, 125, 530, 345, -Math.PI/2.0]);
        var ch = Math.floor(choices.length*Math.random());
        px = Math.floor(Math.random() * (choices[ch][2] - choices[ch][0] + 1)) + choices[ch][0];
        py = Math.floor(Math.random() * (choices[ch][3] - choices[ch][1] + 1)) + choices[ch][1];
        ang = Math.PI/3 * Math.random() - Math.PI/6 + choices[ch][4];
    }
}

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
    if (ML) {
        v = maxv;
    }
    vx = v * Math.cos(ang); // Update velocities and positions
    vy = v * Math.sin(ang);
    px = px + vx;
    py = py + vy;
    
    if (ML == false) { // Draw car now since it wont hurt carrot_stick
        draw();
        draw_car(px, py, ang);
        if (WHISKERS == true) {
            draw_whiskers(get_state(px, py, ang));
        }
    }

    // Check State

    // Kill

    // Give Rewards

    // Update weights idk
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
            throttle = 0;
    }
}
