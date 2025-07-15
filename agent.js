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

/*
ML ==========================================================================
*/
var strategy = new EpsGreedyStrat(expl_rate_start, expl_rate_end, expl_rate_decay);
var agent = new Agent(strategy, num_actions);
var memory = new ReplayMemory(memory_size);
