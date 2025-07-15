var batch_size = 64;
var lr = 0.1; // 0.01
var expl_rate_start = 1;
var expl_rate_end = 0.25;
var expl_rate_decay = 0.01;
var gamma = 0.99;
var discountRate = 0.99;
var episodes = 10;
var max_timesteps = 400;
var memory_size = 256; // replayBufferSize

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
class Experience {
    constructor(state, action, next_state, reward, endrun) {
        this.state = state;
        this.action = action;
        this.next_state = next_state;
        this.reward = reward;
        this.endrun = endrun;
    }
}

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

/*
ML ==========================================================================
*/
var strategy = new EpsGreedyStrat(expl_rate_start, expl_rate_end, expl_rate_decay);
var agent = new Agent(strategy, num_actions);
var memory = new ReplayMemory(memory_size);

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
