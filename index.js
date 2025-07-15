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

function pushGradients(record, gradients) {
    // https://github.com/tensorflow/tfjs-examples/blob/f4b036afbb0f61979da0bfa2dc4a41ceb7d60838/cart-pole/index.js#L237
    for (const key in gradients) {
        if (key in record) {
            record[key].push(gradients[key]);
        } else {
            record[key] = [gradients[key]];
        }
    }
}

window.onload = function(){
    const policyNet = tf.sequential();

    policyNet.add(tf.layers.dense({name: 'dense_0',
                                    inputShape: [whisker_angles.length],
                                    units: 8,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'relu'}));
    policyNet.add(tf.layers.dense({name: 'dense_1',
                                    units: 8,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'relu'}));
    policyNet.add(tf.layers.dense({name: 'dense_2',
                                    units: 3,
                                    kernelInitializer: 'glorotNormal',
                                    activation: 'softmax'}));
    var targetNet = policyNet;
    targetNet.trainable = false;
    // https://stackoverflow.com/questions/48460057/what-does-it-mean-that-a-tf-variable-is-trainable-in-tensorflow/48460190
    policyNet.trainable = true;
    var optimizer = tf.train.adam(lr);

    if (ML == false) {
        setInterval(gameloop, 1000/fps);
    }
    else {
        //========================================BIG ML LOOP===================================
        var timesteps_history = [];
        var px_history = [];
        var py_history = [];
        var timestep=0
        var allGradients = [];
        var allRewards = [];
        var allGameScores = [];
        var cumulativeReward = 0;
        var gameSteps = [];
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
            const gameRewards = []; // 120
            const gameGradients = []; // 121
            var old_state = get_state(px, py, ang);
            var new_state = old_state;
            cumulativeReward = 0;
            for (var timestep=0; timestep < max_timesteps; timestep++) { // 122
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
                gameRewards.push(new_reward); // 146
                px_history.push(px);
                py_history.push(py);
                //var new_reward_tensor = tf.tensor([new_reward]);
                var experience = new Experience(old_state,
                                                action,
                                                new_state,
                                                new_reward,
                                                endrun);
                memory.push(experience);
                cumulativeReward += new_reward;

                if (memory.can_provide_sample(batch_size)) {
                    var experiences = memory.get_rand_memory(batch_size); // https://youtu.be/ewRw996uevM?t=348

                    // https://youtu.be/ewRw996uevM?t=352
                    var batch_states = [];
                    var batch_actions = [];
                    var batch_next_states = [];
                    var batch_old_rewards = [];
                    //var batch_new_rewards = [];
                    var batch_endruns = [];
                    for (var idx = 0; idx < experiences.length; idx++) {
                        batch_states.push(experiences[idx].state);
                        batch_actions.push(parseInt(experiences[idx].action));
                        batch_next_states.push(experiences[idx].next_state);
                        batch_old_rewards.push(experiences[idx].reward);
                        //batch_new_rewards.push(experiences[idx].new_reward);
                        batch_endruns.push(experiences[idx].endrun);
                    }
                    const inputTensor = tf.tensor2d(batch_states);
                    const actionTensor = tf.tensor1d(batch_actions).toInt();
                    const nextStateTensor = tf.tensor2d(batch_next_states);
                    const oldRewardTensor = tf.tensor1d(batch_old_rewards);
                    //const newRewardTensor = tf.tensor1d(batch_new_rewards);
                    const endrunTensor = tf.tensor1d(batch_endruns);


                    // Pseudo Code - https://youtu.be/PyQNfsGUnQA?t=40
                    // Output Q Values... I guess?

                    // https://youtu.be/ewRw996uevM?t=377
                    // policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
                    const currentQTensor = policyNet.predict(inputTensor); // Also needs actions?

                    
                    

                    // https://youtu.be/ewRw996uevM?t=785
                    // Get final states, basically the game over states
                    // For each next state we want the max q value predicted by target net
                    //    among all possible next actions
                    final_states = []; // if an episode is ended by a given action then we call the next
                                        //state that occurs after that action was taken the final state
                                        // We dont want to pass these to targetNet
                    var final_state_locations = [];
                    var non_final_states = []; // https://youtu.be/ewRw996uevM?t=937
                    var non_final_state_locations = [];
                    for (var i = 0; i < batch_size; i++) {
                        if (batch_endruns[i] == 1) {
                            final_states.push(batch_next_states[i]);
                            final_state_locations.push(1);
                            non_final_state_locations.push(0); // https://youtu.be/ewRw996uevM?t=937
                        }
                        else {
                            non_final_states.push(batch_next_states[i]);
                            final_state_locations.push(0);
                            non_final_state_locations.push(1); // https://youtu.be/ewRw996uevM?t=937
                        }
                    }
                    const nonFinalStatesTensor = tf.tensor2d(non_final_states); // https://youtu.be/ewRw996uevM?t=937
                    var values = tf.tensor(new Array(batch_size).fill(0));
                    values[non_final_state_locations] = targetNet.predict(nonFinalStatesTensor)
                        //.max(dim=1)[0].detach();

                    // Get Target Q Values... I guess?
                    


                    // https://youtu.be/ewRw996uevM?t=478
                    //current_q_values = QValues.get_current(policyNet, batch_states, batch_actions);
                    // https://youtu.be/ewRw996uevM?t=753
                    
                    //next_q_values = QValues.get_next(targetNet, batch_next_states);
                    // https://youtu.be/ewRw996uevM?t=781
                    
                    //target_q_values = (next_q_values * gamma) + rewards;
                    //target_q_values = (next_q_values * gamma) + oldRewardTensor; // batch_old_rewards
                    
                        //q*(s,a)=E[Rt+1+gamma max q* (s', a')]

                    //loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1));
                    //optimizer.zero_grad(); // make sure to zero out gradients so you dont sum up
                    //loss.backward();
                    //optimizer.step();
                    //optimizer.applyGradients(
                        //scaleAndAverageGradients(allGradients, normalizedRewards));
                    function getGradientsAndSaveActions() {
                        // https://github.com/tensorflow/tfjs-examples/blob/f4b036afbb0f61979da0bfa2dc4a41ceb7d60838/cart-pole/index.js#L181
                        const f = () => tf.tidy(() => {
                            const nextMaxQTensor = targetNet.predict(nextStateTensor).max(-1);
                    
                            const qs = policyNet.apply(inputTensor, {training: true})
                                .mul(tf.oneHot(actionTensor, num_actions)).sum(-1);
                            current_q_values = values;
                            next_q_values = values;
                            target_q_values = next_q_values.mul(gamma).add(oldRewardTensor);
                            loss = tf.losses.meanSquaredError(current_q_values, target_q_values);
                            return loss;
                        });
                        var varGrads = tf.variableGrads(f);
                        return varGrads;
                    }
                    optimizer.applyGradients(getGradientsAndSaveActions());






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
                    allRewards.push(gameRewards);
                    //draw_drive(px_history, py_history);
                    break;
                }
            } // End of Timesteps
            gameSteps.push(gameRewards.length);
            pushGradients(allGradients, gameGradients);
            allRewards.push(gameRewards);
        } // End of Episodes
    }
    console.log('===========OVER==============');
    console.log('timesteps_history = ', timesteps_history);
    console.log('scores = ', allGameScores);
    draw_drive(px_history, py_history);
    /*
    // Saves the model
    if (Math.max(...timesteps_history) > 100) {
        async function asyncSave() {
            console.log('trying to save');
            const saveResults = await policyNet.save("downloads://my-model-1");
            console.log('after save');
        }
        asyncSave();
    }
    else {
        console.log('no big runs');
    }
    */
};

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
