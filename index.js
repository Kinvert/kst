/*
TODO
    Grayscale noise to create hills
    Better comments to document code
    How to handle edge of map - probably just fill green
    Improve physics
    Sky
    Background, like far away hills etc
    Reduce distortion, straight lines right next to the car seem to curve
    https://deeplizard.com/learn/video/ewRw996uevN
*/

const ML = true;
const ISOMETRIC = false;
//const whisker_angles = [-Math.PI/2, -Math.PI/3, -Math.PI/4, -Math.PI/6, 0,
//                        Math.PI/6, Math.PI/4, Math.PI/3, Math.PI/2];
const whisker_angles = [-Math.PI/4, -Math.PI/6, 0,
                        Math.PI/6, Math.PI/4];
const max_whisker_length = 100; // 200
var batch_size = 64;
var lr = 0.1; // 0.01
var expl_rate_start = 1;
var expl_rate_end = 0.25;
var expl_rate_decay = 0.01;
var gamma = 0.99;
var episodes = 10;
var max_timesteps = 400;
var num_actions = 3;
var memory_size = 256; // replayBufferSize

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
    get_rand_memory(size) {
        var arr = this.memory;
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
        return this.memory.length >= batch_size;
    }

    num_memories() {
        return this.memory.length;
    }
}

// //https://www.youtube.com/watch?v=PyQNfsGUnQA
class Experience {
    constructor(state, action, next_state, reward) {
        this.state = state;
        this.action = action;
        this.next_state = next_state;
        this.reward = reward;
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

/*
function extract_tensors(experiences) {
    // https://youtu.be/ewRw996uevM?t=607
    console.log(experiences);
    //batch = Experience(*zip(*experiences))

    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)

    return {states: states, actions: actions, rewards: rewards, next_states: next_states}
}
*/

class QValues {
    // https://youtu.be/ewRw996uevM?t=730
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
canv.focus()
var ctx = canv.getContext("2d");
var canv2 = document.getElementById("myCanvas2");
canv2.setAttribute('tabindex', '1')
canv2.focus()
var ctx2 = canv2.getContext("2d");
if (ISOMETRIC) {
    var ctx2 = canv2.getContext("2d");
}
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

function draw_whiskers() {
    ctx.strokeStyle = 'magenta';
    ctx.lineWidth = 2;
    var {whisker_ends} = get_whisker_ends(px, py, ang);
    for ( var ww = 0; ww < whisker_angles.length*2; ww += 2) {
        ctx.moveTo(px, py);
        ctx.lineTo(whisker_ends[ww], whisker_ends[ww+1]);
        ctx.stroke();
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
            if (red = 69 && green == 142 && blue > 49 && blue < 55) {
                hit_green = true;
                break;
            }
        }
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
    //var tensor = tf.tensor([whisker_lengths]);
    //return tf.reshape(tensor, [-1, whisker_angles.length, 1]);

    return whisker_lengths;

    //var tensor = tf.tensor2d([whisker_lengths]);
    //return tensor;
}

function get_state_old() {
    /* This was for feeding in an image
    var imgForTensor = ctx2.getImageData(0, horiz, W, H-horiz)
    var tensor = tf.browser.fromPixels(imgForTensor);
    return tf.reshape(tensor, [-1, horiz, W, 3]);
    */
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
    if ( action == 0 && action == 2) {
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



window.onload = function(){
    console.log('WINDOW ON LOAD');
    const policyNet = tf.sequential();
    //policyNet.add(tf.layers.conv2d({inputShape: [H/2, W, 3], // FROM IMAGES
    /*
    //policyNet.add(tf.layers.conv1d({inputShape: [whisker_angles.length, 1], // WORKED BEFORE BATCH
                                    name: 'conv1d_1',
                                    filters: 25,
                                    kernelSize: 5,
                                    padding: 'same',
                                    activation: 'relu'}));
    */
    policyNet.add(tf.layers.dense({name: 'dense_0',
                                    inputShape: [whisker_angles.length],
                                    units: 8,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'relu'}));
    //policyNet.add(tf.layers.flatten());
    policyNet.add(tf.layers.dense({name: 'dense_1',
                                    units: 8,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'relu'}));
    policyNet.add(tf.layers.dense({name: 'dense_2',
                                    units: 3,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'softmax'}));
    var targetNet = policyNet; // This isn't for training this is for evaluating
    targetNet.trainable = false;
    policyNet.trainable = true;
    var optimizer = tf.train.adam(lr);

    if (ML == false) {
        setInterval(gameloop, 1000/fps);
    }
    else {
        /*
        ======================================================================================
        ======================================================================================
        ========================================BIG ML LOOP===================================
        ======================================================================================
        ======================================================================================
        */
        var timesteps_history = [];
        var px_history = [];
        var py_history = [];
        var timestep=0
        var allGradients = [];
        var allRewards = [];
        var allGameScores = [];
        var cumulativeReward = 0;
        for (var eps = 0; eps < episodes; eps += 1) {
            console.log(' ');
            console.log(' ');
            console.log('GAME=', eps,
                '   steps=', timestep,
                '   score=', cumulativeReward.toFixed(3));
            game_reset()
            //agent.reset_steps();
            var endrun = 0;
            var new_reward = 0;
            const gameGradients = [];
            const gameRewards = [];
            var old_state = get_state(px, py, ang);
            var new_state = old_state;
            cumulativeReward = 0;
            for (var timestep=0; timestep < max_timesteps; timestep++) {
                var old_state_tensor = make_state_tensor(old_state);
                var action = agent.select_action(tf.tensor2d([old_state]), policyNet);
                do_the_thing(action); // Take action
                gameloop(); // Run the timestep through super physics
                old_state = new_state;
                var new_state = get_state(px, py, ang);
                old_reward = new_reward;
                new_reward = carrot_stick(px, py, action);
                if (new_reward == -1) {
                    endrun = 1;
                }
                allRewards.push(new_reward);
                gameRewards.push(new_reward);
                px_history.push(px);
                py_history.push(py);
                var new_reward_tensor = tf.tensor([new_reward]);
                var experience = new Experience(old_state,
                                                action,
                                                new_state,
                                                new_reward_tensor,
                                                endrun);
                memory.push(experience);
                cumulativeReward += new_reward;
                if (memory.can_provide_sample(batch_size)) {
                    var experiences = memory.get_rand_memory(batch_size);
                    var batch_states = [];
                    var batch_actions = [];
                    //var batch_next_states = [];
                    //var batch_old_rewards = [];
                    //var batch_new_rewards = [];
                    //var batch_endruns = [];
                    for (var idx = 0; idx < experiences.length; idx++) {
                        batch_states.push(experiences[idx].state);
                        batch_actions.push(parseInt(experiences[idx].action));
                        //batch_next_states.push(experiences[idx].next_state);
                        //batch_old_rewards.push(experiences[idx].old_reward);
                        //batch_new_rewards.push(experiences[idx].new_reward);
                        //batch_endruns.push(experiences[idx].endrun);
                    }
                    const stateTensor = tf.tensor2d(batch_states);
                    const actionTensor = tf.tensor1d(batch_actions).toInt();
                    //const nextStateTensor = tf.tensor2d(batch_next_states);
                    //const oldRewardTensor = tf.tensor1d(batch_old_rewards);
                    //const newRewardTensor = tf.tensor1d(batch_new_rewards);
                    //const endrunTensor = tf.tensor1d(batch_endruns);

                    // Pseudo Code - https://youtu.be/PyQNfsGUnQA?t=40
                    // Output Q Values... I guess?
                    //const currentQTensor = policyNet.predict(stateTensor);

                    
                    

                    // https://youtu.be/ewRw996uevM?t=829
                    // Get final states, basically the game over states
                    /*
                    final_states = [];
                    final_state_locations = [];
                    non_final_states = [];
                    non_final_state_locations = [];
                    for (var b = 0; b < batch_size; b++) {
                        if (batch_endruns[b] == 1) {
                            final_states.push(batch_next_states[b]);
                            final_state_locations.push(1);
                            non_final_state_locations.push(0);
                        }
                        else {
                            non_final_states.push(batch_next_states[b]);
                            final_state_locations.push(0);
                            non_final_state_locations.push(1);
                        }
                    }
                    */
                    //var values = tf.tensor(new Array(batch_size).fill(0));
                    //values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()








                    // Get Target Q Values... I guess?
                    //const nextMaxQTensor = targetNet.predict(nextStateTensor).max(-1);
                    
                    //const qs = policyNet.apply(stateTensor, {training: true})
                    //    .mul(tf.oneHot(actionTensor, num_actions)).sum(-1);


                    // https://youtu.be/ewRw996uevM?t=478
                    //current_q_values = QValues.get_current(policyNet, states, actions)
                    //next_q_values = QValues.get_next(target_net, next_states)
                    //target_q_values = (next_q_values * gamma) + rewards
                        //q*(s,a)=E[Rt+1+gamma max q* (s', a')]

                    //loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                    //optimizer.zero_grad() make sure to zero out gradients so you dont sum up
                    //loss.backward()
                    //optimizer.step()

                    //if em.done
                        //episode_durations.append(timestep)
                        //plot episode_durations, 100) this is related to how long balance rod
                        //break

                    //if episode % target_update == 0
                        //target_net.load_state_dict(policyNet.state_dict())

                    //const inputTensor = cartPoleSystem.getStateTensor();
                
                    // Get the probability of the leftward action.
                    //const leftProb = tf.sigmoid(logits);
                    // Probabilites of the left and right actions.
                    //const leftRightProbs = tf.concat([leftProb, tf.sub(1, leftProb)], 1);
                    //const actions = tf.multinomial(leftRightProbs, 1, null, true);
                    //return [logits, actions];

                    //currentActions_ = actions.dataSync();

                    //const labels = f.sub(1, actionsTensor);

                    //tf.losses.sigmoidCrossEntropy(labels, logits).asScalar(); // LOSSES
                    //var thing = tf.variableGrads(f);


                    //thing.grads;





                    /*
                    // https://chat.stackoverflow.com/transcript/191041/2019/4/1/17-21
                    //function train(actions, rewards, boards) {
                    function train(actionTensor, rewardTensor, stateTensor, model) {
                        //const optimizer = tf.train.rmsprop(this.learning_rate, 0.99)
                        var optimizer = tf.train.adam(lr);
                        //const oneHotLabels = tf.oneHot(actions, BOARD_SIZE)
                        const oneHotLabels = tf.oneHot(actionTensor, 3);
                        //https://12ft.io/proxy?q=https://www.geeksforgeeks.org/tensorflow-js-tf-train-optimizer-class-minimize-method
                        optimizer.minimize(() => {
                            //const logits = model.predict(tf.tensor(boards))
                            const logits = model.predict(stateTensor);
                            //const crossEntropies = tf.losses.softmaxCrossEntropy(oneHotLabels, logits)
                            const crossEnts = tf.losses.softmaxCrossEntropy(oneHotLabels, logits);
                            //const loss = tf.sum(tf.tensor(rewards).mul(crossEntropies)).asScalar()
                            const loss = tf.sum(rewardTensor.mul(crossEnts)).asScalar();
                            return loss;
                        })
                    }
                    train(actionTensor, oldRewardTensor, stateTensor, policyNet);
                    // END https://chat.stackoverflow.com/transcript/191041/2019/4/1/17-21
                    */








                    // CARTPOLE METHOD
                    //var one_hot_prep = [];
                    //for (var i = 0; i < batch_size; i++) {
                    //    one_hot_prep.push(make_action_one_hot(batch_actions[i]));
                    //}
                    //var one_hot_tensor = tf.tensor2d(one_hot_prep);

                    function getGradientsAndSaveActions(sT) {
                        const f = () => tf.tidy(() => {

                            const logits = policyNet.predict(sT);

                            const actions_one_hot = tf.oneHot(actionTensor, 3);
                            this.currentActions = actions_one_hot.dataSync();
  
                            var labels = tf.sub(1,
                                tf.tensor2d(this.currentActions, actions_one_hot.shape));

                            var loss = tf.losses.sigmoidCrossEntropy(labels, logits).asScalar();
                            return loss;
                        });
                        //const {vValue, vGrads} = tf.variableGrads(f);
                        var varGrads = tf.variableGrads(f);
                        return varGrads;
                    }

                    const gradients = tf.tidy(() => { // https://github.com/tensorflow/tfjs-examples/blob/master/cart-pole/index.js#L126
                        return getGradientsAndSaveActions(stateTensor);
                    });
                    // https://github.com/tensorflow/tfjs-examples/blob/master/cart-pole/index.js#L131
                    var asdf = gradients;
                    optimizer.applyGradients(asdf.grads);
                   //END CARTPOLE




                    // For Standfords Version: https://youtu.be/lvoHnicueoE?t=1551
                    // https://towardsdatascience.com/practical-guide-for-dqn-3b70b1d759bf







                    

                    
                    
                    //const doneMask = tf.scalar(1).sub(
                    //    tf.tensor1d(experiences.map(example => example[3])).asType('float32'));

                    //const targetQs =
                    //    newRewardTensor.add(nextMaxQTensor.mul(doneMask).mul(gamma));

                    //var losses_thing =  tf.losses.meanSquaredError(targetQs, qs);
                    
                } // End of memory batch size thing
                if (endrun == 1) {
                    console.log('            GAME OVER MAN GAME OVER');
                    timesteps_history.push(timestep);
                    allGameScores.push(cumulativeReward);
                    //draw_drive(px_history, py_history);
                    break;
                }
            }
        }
    }
    console.log('===========OVER==============');
    console.log('timesteps_history = ', timesteps_history);
    console.log('scores = ', allGameScores);
    draw_drive(px_history, py_history);
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
    
    if (ML == false) { // Draw car now since it wont hurt carrot_stick
        draw();
        draw_car(px, py, ang);
        draw_whiskers(get_state(px, py, ang));
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
    ctx.lineWidth = 1;
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
    //draw_tiny_top_view();

    if (ISOMETRIC) {
        draw_mode_7();
    }

    // TODO Difference between this img and last img to only show changed pixels
}

function draw_top_view() {
    /* TINY TOP VIEW */
    /*
    ctx2.save()
    ctx2.rotate(-Math.PI/2-ang);
    ctx2.drawImage(canv, -flx, -fly);
    ctx2.restore();
    */
}

function draw_mode_7() {
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