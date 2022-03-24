// CARTPOLE METHOD
                // TODO MAYBE A BETTER WAY
                //https://stackoverflow.com/questions/55444995/trouble-with-training-simple-policy-agent-error-cannot-find-a-connection-betwe

                function getGradientsAndSaveActions(inputTensor) {
                    // https://github.com/tensorflow/tfjs-examples/blob/f4b036afbb0f61979da0bfa2dc4a41ceb7d60838/cart-pole/index.js#L181
                    const f = () => tf.tidy(() => {
                        const logits = policyNet.predict(inputTensor);
                        const actions_one_hot = tf.oneHot(actionTensor, 3);
                        this.currentActions = actions_one_hot.dataSync();
                        var labels = tf.sub(1,
                            tf.tensor2d(this.currentActions, actions_one_hot.shape));

                        var loss = tf.losses.sigmoidCrossEntropy(labels, logits).asScalar();
                        return loss;
                    });
                    var varGrads = tf.variableGrads(f);
                    return varGrads;
                }

                const gradients = tf.tidy(() => { // 126
                    // https://github.com/tensorflow/tfjs-examples/blob/master/cart-pole/index.js#L126
                    return getGradientsAndSaveActions(inputTensor);
                });
                // https://github.com/tensorflow/tfjs-examples/blob/master/cart-pole/index.js#L131
                
                //var asdf = gradients;
                //optimizer.applyGradients(asdf.grads);

                pushGradients(gameGradients, gradients); // 131
                //const action = this.currentActions_[0]; //132
                const isDone = endrun; // 133


                function discountRewards(rewards, discountRate) {
                    // https://github.com/tensorflow/tfjs-examples/blob/f4b036afbb0f61979da0bfa2dc4a41ceb7d60838/cart-pole/index.js#L332
                    const discountedBuffer = tf.buffer([rewards.length]);
                    let prev = 0;
                    for (let i = rewards.length - 1; i >= 0; --i) {
                        const current = discountRate * prev + rewards[i];
                        discountedBuffer.set(current, i);
                        prev = current;
                    }
                    return discountedBuffer.toTensor();
                }

                function discountAndNormalizeRewards(rewardSequences, discountRate) {
                    // https://github.com/tensorflow/tfjs-examples/blob/f4b036afbb0f61979da0bfa2dc4a41ceb7d60838/cart-pole/index.js#L358
                    return tf.tidy(() => {
                        const discounted = [];
                        console.log('rewardSequences', rewardSequences);
                        for (const sequence of rewardSequences) {
                            console.log('sequence', sequence);
                            discounted.push(discountRewards(sequence, discountRate))
                        }
                        // Compute the overall mean and stddev.
                        const concatenated = tf.concat(discounted);
                        const mean = tf.mean(concatenated);
                        const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
                        // Normalize the reward sequences using the mean and std.
                        const normalized = discounted.map(rs => rs.sub(mean).div(std));
                        return normalized;
                    });
                }

                function scaleAndAverageGradients(allGradients, normalizedRewards) {
                    // https://github.com/tensorflow/tfjs-examples/blob/f4b036afbb0f61979da0bfa2dc4a41ceb7d60838/cart-pole/index.js#L389
                    /*
                    console.log(' ');
                    console.log(' ');
                    console.log('allGradients', allGradients);
                    console.log('allGradients.length', allGradients.length);
                    console.log('normalizedRewards', normalizedRewards);
                    return tf.tidy(() => {
                        const gradients = {};
                        for (const varName in allGradients) {
                            gradients[varName] = tf.tidy(() => {
                                // Stack gradients together.
                                const varGradients = allGradients[varName].map(
                                    varGameGradients => tf.stack(varGameGradients));
                                // Expand dimensions of reward tensors to prepare for
                                // multiplication with broadcasting.
                                const expandedDims = [];
                                for (let i = 0; i < varGradients[0].rank - 1; ++i) {
                                    expandedDims.push(1);
                                }
                                const reshapedNormalizedRewards = normalizedRewards.map(
                                    rs => rs.reshape(rs.shape.concat(expandedDims)));
                                console.log('varGradients.length', varGradients.length);
                                for (let g = 0; g < varGradients.length; ++g) {
                                    console.log('g', g);
                                    // This mul() call uses broadcasting.
                                    console.log('varGradients[g]', varGradients[g]);
                                    console.log('reshapedNormalizedRewards[g]',
                                        reshapedNormalizedRewards[g]);
                                    varGradients[g] = varGradients[g].mul(
                                        reshapedNormalizedRewards[g]);
                                }
                                // Concatenate the scaled gradients together, then average them
                                // across all the steps of all the games.
                                return tf.mean(tf.concat(varGradients, 0), 0);
                            });
                        }
                        return gradients;
                    });
                    */
                    return allGradients;
                }

                tf.tidy(() => {
                    // The following line does three things:
                    // 1. Performs reward discounting, i.e., make recent rewards count more
                    //    than rewards from the further past. The effect is that the reward
                    //    values from a game with many steps become larger than the values
                    //    from a game with fewer steps.
                    // 2. Normalize the rewards, i.e., subtract the global mean value of the
                    //    rewards and divide the result by the global standard deviation of
                    //    the rewards. Together with step 1, this makes the rewards from
                    //    long-lasting games positive and rewards from short-lasting
                    //    negative.
                    // 3. Scale the gradients with the normalized reward values.
                    const normalizedRewards =
                        discountAndNormalizeRewards(allRewards, discountRate);
                    // Add the scaled gradients to the weights of the policy network. This
                    // step makes the policy network more likely to make choices that lead
                    // to long-lasting games in the future (i.e., the crux of this RL
                    // algorithm.)
                    optimizer.applyGradients(
                        scaleAndAverageGradients(allGradients, normalizedRewards));
                });
                console.log('DISPOSE allGradients');
                tf.dispose(allGradients);
                //return gameSteps;
                //END CARTPOLE