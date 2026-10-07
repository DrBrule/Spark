#!/usr/bin/env python
# coding: utf-8

# <link rel="stylesheet" type="text/css" href="./style.css">
# 
# <div class="tutorial-header">
# 
# ## Tutorial \#6: Solving the CartPole™
# 
# </div>

# <link rel="stylesheet" type="text/css" href="./style.css">
# 
# <div class="tutorial-text">
# 
# Now it is the time to wrap all that we have learned so far and use <b><i>Spark</i></b> to solve one of the most iconic problems in reinforcement learning: the CartPole™.
# 
# The [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) is a simple control problem in which a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.
# 
# In order to solve the network, we require a neuron model capable of not only processing spikes, but also capable of receiving a signal like a reward in order to modulate its synaptic plasticity dynamics. For this case we will use a simple ALIF neruon (Adaptive-Leaky-Integrate-and-Fire). We will also add a modulated versions of a classical plasticity protocol known as the "quadruplet rule", which we are going to populate with parameters from: [Plasticity Cards®](https://plasticity.vogelslab.org/), a collection of parameters for the quadruplet rule that tends to produce stable dynamics. We took the liberty to prepare this neuronal model in advance (which will later appear as <b>spark.ALIFModNeuron</b>) and pack it into a file!.
# 
# Considering the action space for the CartPole problem, the brain needs to be capable on picking two actions: "move left" or "move right" depending on the current state of the cart. The input to the problem are four bounded float values: position, velocity, pole angle and pole angular velocity. Given the very nature of these values (they are continuous values that may be represented by segments of a line), we can use a <b>spark.nn.interfaces.TopologicalLinearSpiker</b> to map these values to an spiking pattern along a line (akin to grid cells!). For the core of the network, an "A vs B" design pattern will naturally induce a choice of left or right. This pattern may be summarized as: each population tries to main a high level of activity and supress the activity of the other population if it "believes" that it needs to act. Finally, for the output of the network, we can simply take the activity of A and B and pick which ever is more active at the present moment. In order to accomplish this, we will once again use the <b>spark.nn.interfaces.ExponentialIntegrator</b> and pick the action associated with most active population.
# 
# Overall, the plan is:
# 
# 1. Design a neuron model that may be modulated via rewards.
# 2. Assamble everything into a brain.
# 3. Train the model
# 4. Profit! 
# 
# For this tutorial we prepared two files: <b>alif_mod_neuron.scfg</b> and <b>ab_brain.scfg</b> for the neuron and the brain respectively. You can find these two files in the main repository under the directory <b>/tutorials</b>.
# 
# Let's start by looking at the neuron model.
# 
# </div>

# In[1]:


import spark
# Imports
# get_ipython().run_line_magic('gui', 'qt')
import jax
import spark
import numpy as np
import jax.numpy as jnp

NEURON_PATH = 'tutorials/alif_mod_neuron.scfg'
BRAIN_PATH = 'tutorials/ab_brain.scfg'


# editor = spark.GraphEditor()
# 
# editor.launch()
# 
# editor.open_model(NEURON_PATH)
# editor.open_model(BRAIN_PATH)

# 
# # Feel free to explore the neuron model!
# if __name__ == "__main__":
# 


# # We can either register the neuron directly from the file
# try:
# 	spark.register_neuron_from_config_file('ALIFModNeuron', NEURON_PATH)
# except Exception as e:
# 	print(f'(File) Oh no: {e}!')

# Note that after this simple instruction, the ALIFModNeuron model is now available in the editor both, as a compact reference when building a brain or as a full editable model when building a Neuron!. Furthermore, we can also inspect the brain model directly and tune it's parameters.


try:
	brain_config = spark.nn.BrainConfig.from_file(BRAIN_PATH)
	print(f'Infinite success!')
except Exception as e:
	print(f'Oh no: {e}!')




# You may also tinker with the brain model!
#if __name__ == "__main__":
#    editor.open_model(BRAIN_PATH)
# 
# # With this, we are now ready to build the brain and attack the 
# CartPole problem!
# # 
# # As usual, we need to define the functions to run the
#  model efficiently. Up to this point we have been making single calls to the model. However, this approach will quickly become highly innefficient, since SNN are expected to integrate the output, diggest it for some time and then output an answer (ideally whenever they managed to get the answer, but this is easier said than done).
# # 
# # In <b><i>Spark</i></b>, we refer to this concept as 
# "interaction times", which is simply the number of 
# time steps we let the model process the input before 
# trying to retrieve an answer. There is two ways to 
# implement this, with loops as we saw in the one of 
# the first tutorials or by unrolling (the Jax™ way).
# # 
# # The interaction time is also what makes the modulatory 
# signal interesting. A reward is delivered once, by the 
# environment, but the network keeps ticking for the 
# whole interaction window and the reward has to 
# remain meaningful for some of it. That decay is 
# part of the model, not of the training loop, 
# so the neuron carries a 
# <b>spark.nn.interfaces.SignalTrace</b> 
# node: the modulation port feeds the trace, 
# and the plasticity rule reads the trace 
# rather than the raw input. All the caller 
# decides is <i>when</i> a signal reaches 
# the network. A reward is an <b>event</b>, 
# delivered once and carried by the trace; the small 
# passive drift is a <b>sustained</b> background, 
# delivered on every step, and the trace settles on it.
# # 
# # </div>



from functools import partial

@partial(jax.jit, static_argnames=['steps'])
def run_model_loop(graph, state, steps, **inputs):
    # Merge model
    model = spark.merge(graph, state)
    # The reward reaches the network once; the drift is present on every step. Neither is
    # decayed here: the traces inside the neuron own that.
    reward_keys = ('mod_a_ex', 'mod_a_in', 'mod_b_ex', 'mod_b_in')
    silence = {key: spark.FloatArray(jnp.zeros_like(inputs[key].value)) for key in reward_keys}
    # Run model
    for step in range(steps):
        out = model(**inputs)
        inputs = {**inputs, **silence}
    # Get new state
    _, state = spark.split((model))
    return out, state



@partial(jax.jit, static_argnames=['steps', 'unroll'])
def run_model_unroll(graph, state, steps, unroll=15, **inputs):
    # The reward reaches the network once and its trace carries it through the rest of the
    # interaction window; the drift is present on every step and its trace settles on it.
    # Neither decay lives here any more, so nothing in this function has to agree with dt.
    reward_keys = ('mod_a_ex', 'mod_a_in', 'mod_b_ex', 'mod_b_in')
    delivered_once = jnp.zeros((steps,), dtype=jnp.float32).at[0].set(1.0)
    rewards = {
        key: (inputs[key].value.astype(jnp.float32)[None, :] * delivered_once[:, None]).astype(jnp.float16) for key in reward_keys
	}
    # Everything else, the drift included, is held across the window
    held_inputs = {
        key: value for key, value in inputs.items() if key not in reward_keys
	}
    def step_fn(model_state, reward_slice):
        # Merge model
        model = spark.merge(graph, model_state)
        # Run model. Append the reward to inputs.
        out = model(**held_inputs, **{key: spark.FloatArray(value) for key, value in reward_slice.items()})
        # Get new state
        _, new_state = spark.split((model))
        return new_state, out
    # Run the scan loop
    final_state, stacked_outs = jax.lax.scan(
        step_fn, 
        state, 
        xs=rewards, 
        length=steps, 
        unroll=unroll
	)
    # Retrieve model output
    final_out = jax.tree.map(lambda x: x[-1], stacked_outs)
    return final_out, final_state


# <link rel="stylesheet" type="text/css" href="./style.css">
# 
# <div class="tutorial-text">
# 
# Although both, the loop and the unroll approaches, 
# will produce the equivalent simulations the actual 
#computation performed is slightly different. 
#The difference is just how the final computational 
#graph is seen by the JIT compiler and how it is 
#optimized. The for loop exposes "everything" 
#to JIT, while the unroll just a fraction of 
#that "everything", which is them repeated over 
#and over again. The end result is that the for 
#loop compilation may take a really long time, 
#while unrolling the computation is considerably 
#faster. However, unrolling the computation may 
#produce a slightly less performant call to the 
#function. Note that, how much is unrolled may be 
#tweaked by the number of unrolling steps. 
#An unrolling of one step will compile fast 
#but may not be as performant as a unrolling 
#of 15 steps, which requires considerably more 
#compilation time. For this particular example, 
#1 vs 15 unrolling steps results in a speed gain 
#of ~250μs but the compilation time goes from 
#~0.7s to ~9.7s. For long runs, the 250μs 
#will acumulate faster than whatever we can 
#save in compilation times. Setting the number 
#of unrolling equal to the toal steps will 
#produce a program equivalent to the for loop. So, whatever floats your SNN.
# 
# Now, as usual in any reinforcement learning setup, 
#we need to create a training loop.
# </div>

# In[10]:


# Reward metaparameters
# NOTE: No correction factors. The reward is delivered once and read by a SignalAccumulator,
# the passive drift is delivered on every step and read by a SignalAverage, and each trace
# keeps its own signal at its own scale.
PASSIVE_MODULATION_SCALE = -0.0001
REWARD_FALL_SCALE = -33
REWARD_OOB_SCALE = -33
REWARD_OOB_THRESHOLD = 0.05
NEG_REWARD_SCALE = -0.1
MOD_INHIBITION_SCALE= 0.3

# Model metaparameters
DT = 1
TRAIN_EPOCHS = 200
UNROLL = 1			# Fast compile, "slow" execution
#UNROLL = 15		# Slow compile, "fast" execution
STEPS = int(50/DT)


# In[11]:


# Utility function to preprocess information from the environment.
# This function just scale the observation environment to the (0,1) range.
def process_obs(x):
    # CartPos, CartSpeed, PoleAngle, PoleAngSpeed 
    x = x / np.array([2.4, 2.5, 0.2095, 3.5])
    x = np.clip(x, a_min=-1, a_max=1)
    return x



def mod_inputs(signal, drift, mod_a=0.0, mod_b=0.0):
	"""Six ports: a reward per population, and one drift per sign shared by both."""
	f16 = lambda v: spark.FloatArray(jnp.array([v], dtype=jnp.float16))
	return {
		'signal': spark.FloatArray(jnp.array(signal, dtype=jnp.float16)),
		'mod_a_ex': f16(mod_a),
		'mod_a_in': f16(mod_a * MOD_INHIBITION_SCALE),
		'mod_b_ex': f16(mod_b),
		'mod_b_in': f16(mod_b * MOD_INHIBITION_SCALE),
		'drift_ex': f16(drift),
		'drift_in': f16(drift * MOD_INHIBITION_SCALE),
	}



def run_episode(env, graph, state, modulation_scale, steps, unroll):
	# Reset environment
	next_obs, _ = env.reset()
	next_obs = process_obs(next_obs)
	# Loop until the episode ends, either from failing or truncation
	terminated = False
	truncated = False
	episode_steps = 0
	# Allow the network to prepare for the episode.
	for _ in range(1):
		_, state = run_model_unroll(
			graph, state, steps=steps, unroll=unroll,
			**mod_inputs(np.zeros_like(next_obs), drift=0.0)
		)
	# Start episode
	while not (terminated or truncated):
		# Model logic, let the network do what it wants to do.
		out, state = run_model_unroll(
			graph, state, steps=steps, unroll=unroll,
			**mod_inputs(next_obs, drift=PASSIVE_MODULATION_SCALE * modulation_scale)
		)
		# Environment logic.
		next_action = int(np.argmax(out['action'].value))
		prev_episode = np.array(next_obs)
		next_obs, _, terminated, truncated, _ = env.step(next_action)
		next_obs = process_obs(next_obs)
		episode_steps += 1
	# Steer the network to what we want with one extra step with the last episode and the reward
	if not truncated:
		# Check if the agent went out of bounds.
		out_of_bounds = np.abs(next_obs[0]) >= 1
		# Critic-like action update
		# Going out of bounds conflicts with learning to balance the pole, adding a second condition
		# to check for the the pole to be mostly vertical improves learning speed 
		if out_of_bounds and np.abs(next_obs[2]) < REWARD_OOB_THRESHOLD:
			# If the cart went out of bounds, we need to move the cart towards the center
			# Going out of bounds is not as bad as the pole falling, so we should not put to much 
			# punishment towards that action
			steer_to_b = next_obs[0] < 0
			reward = REWARD_OOB_SCALE * modulation_scale
		else: 
			# If the pole fell, we need to move the cart towards the side it fell
			steer_to_b = next_obs[2] > 0
			reward = REWARD_FALL_SCALE * modulation_scale
		mod_a = reward if steer_to_b else NEG_REWARD_SCALE * reward
		mod_b = reward if not steer_to_b else NEG_REWARD_SCALE * reward
		# The reward is delivered once. The accumulator inside the neuron carries it through the
		# rest of the interaction window.
		for _ in range(1):
			_, state = run_model_unroll(
				graph, state, steps=steps, unroll=unroll,
				**mod_inputs(prev_episode, drift=0.0, mod_a=mod_a, mod_b=mod_b)
			)
	else:
		out_of_bounds = False
	return state, episode_steps, terminated, truncated, out_of_bounds


# 
# Now, let's put all together and train the brain.
# 
# First, we create and configure the environment. 
# Usually, the cartpole uses a continuous reward of 
# one to provide continuous feedback to ANN models. 
# We instead we are going to use a single -1 
# (which we will post-process) when the episode ends.
# 

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

# Initialize the environment
env_name =  'CartPole-v1'
env = gym.make(env_name, sutton_barto_reward=True)
next_obs, _ = env.reset(seed=412)
next_obs = process_obs(next_obs)


# <link rel="stylesheet" type="text/css" href="./style.css">
# 
# <div class="tutorial-text">
# 
# As usual, we create a Brain, using the pre-defined configuration file and initialize it.
# 
# It is possible to generate a new seed for the model by simply calling <b>config.with_new_seeds</b>, rather than manually changing all seeds. However, this particular model is somewhat sensitive to kernel initialization, as such, final behavior may differ from seed to seed. For the sake of illustration, the default model contains one of the "good seeds". 
# 
# </div>

# In[13]:


# Load the brain configuration
brain_config = spark.nn.BrainConfig.from_file(BRAIN_PATH)

# Reseed
# NOTE: this model is sensitive to kernel initialization, so the tutorial pins a seed
# known to train well. Adding or removing a module shifts how seeds are handed out, which
# is why this number changed when the modulation trace was introduced.
brain_config = brain_config.with_new_seeds(seed=412)

# Initialize the brain
brain = spark.nn.Brain(config=brain_config)
brain(
    signal=spark.FloatArray(jnp.array(next_obs, dtype=jnp.float16)),
    mod_a_ex=spark.FloatArray(jnp.array([0], dtype=jnp.float16)),
    mod_a_in=spark.FloatArray(jnp.array([0], dtype=jnp.float16)),
    mod_b_ex=spark.FloatArray(jnp.array([0], dtype=jnp.float16)),
    mod_b_in=spark.FloatArray(jnp.array([0], dtype=jnp.float16)),
    drift_ex=spark.FloatArray(jnp.array([0], dtype=jnp.float16)),
    drift_in=spark.FloatArray(jnp.array([0], dtype=jnp.float16)),
)
# Split brain into the graph and the state
graph, state = spark.split((brain))


# In[14]:


import tqdm

# Some classic metrics
train_steps = []
ema_alpha, train_steps_ema = 0.8, []
current_train_ema_steps = 0
train_oob = []

# A classic training loop.
for e in (pbar := tqdm.tqdm(range(TRAIN_EPOCHS))):
	# We scale the passive modulation in order to prevent a bad episode from drastically updating the network
	modulation_scale = (1 - (current_train_ema_steps / 500))**(0.5)
	# Run the episode
	state, episode_steps, terminated, truncated, out_of_bounds = run_episode(
		env, 
		graph, 
		state, 
		modulation_scale=modulation_scale,
		steps=STEPS, 
		unroll=UNROLL, 
	)
	# Update the classic metrics
	current_train_ema_steps = train_steps_ema[-1] if len(train_steps_ema) > 0 else 0
	train_steps.append(episode_steps)
	train_steps_ema.append(ema_alpha * current_train_ema_steps + (1-ema_alpha) * episode_steps)
	pbar.set_description(f'Train Steps EMA: {train_steps_ema[-1]:.2f}')
	if out_of_bounds:
		train_oob.append(e)


# <link rel="stylesheet" type="text/css" href="./style.css">
# 
# <div class="tutorial-text">
# 
# Finally, let's visualize how does the agent performed!
# </div>

# In[15]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Process data
s = np.arange(TRAIN_EPOCHS)
ts = np.array(train_steps)
x_oob = np.array(train_oob)
y_oob = ts[x_oob] if len(x_oob) > 0 else []
x_trnc = np.argwhere(ts == 500).reshape(-1)
y_trnc = 500*np.ones_like(x_trnc)
x_pole = [x for x in range(TRAIN_EPOCHS) if (x not in x_oob and x not in x_trnc)]
y_pole = ts[x_pole]

fig = plt.figure(figsize=(10,4))
# General curve
plt.plot(s, train_steps, alpha=1/3, linestyle=':')
plt.plot(s, train_steps_ema, color='tab:purple', alpha=0.8)
# Pole-fell
plt.scatter(x_pole, y_pole, color='tab:red', s=20, alpha=1, marker='x')
# Agent out-of-bounds
plt.scatter(x_oob, y_oob, color='tab:orange', s=20, alpha=1, marker='v')
# Episode truncated
plt.scatter(x_trnc, y_trnc, color='tab:green', s=20, alpha=1, marker='*')

# Format plot
plt.xlim(-5, TRAIN_EPOCHS+5)
plt.ylim(-5, 505)
plt.xlabel('Episode', fontweight='bold', fontsize=12)
plt.ylabel('# Steps', fontweight='bold', fontsize=12)
plt.grid(alpha=0.2)
plt.tight_layout()

legend_elements = [
	Line2D([], [], color='tab:red', marker='v', linewidth=0, label='Pole Fell'), 
	Line2D([], [], color='tab:orange', marker='x', linewidth=0, label='Out of Bounds'), 
	Line2D([], [], color='tab:green', marker='*', linewidth=0, label='Truncation'), 
]
fig.legend(handles=legend_elements, ncols=1, bbox_to_anchor=(0.165, 0.825), loc='center', frameon=True)
plt.show()


# <link rel="stylesheet" type="text/css" href="./style.css">
# 
# <div class="tutorial-text">
# 
# 🎉 Congratulations! (again) 🎉 
# 
# Now you have trained your first SNN that actually does something midly interesting!
# 
# </div>


editor = spark.GraphEditor()

editor.launch()
