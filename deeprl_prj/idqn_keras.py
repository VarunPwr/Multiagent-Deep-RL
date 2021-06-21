'''Keras DQN Agent implementation. Includes Basic Dueling Double DQN and Temporal Attention DQN.'''
# Works on Tf2.x

from deeprl_prj.policy import *
from deeprl_prj.objectives import *
from deeprl_prj.preprocessors import *
from deeprl_prj.utils import *
from deeprl_prj.core import *

import keras
from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Concatenate,
        Permute, Multiply, Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute, multiply)
from keras.models import Model
from keras import backend as K


import sys
from gym import wrappers
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import numpy as np

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.compat.v1.Session(config=config))


def create_model(input_shape, policy_shape, num_actions, mode, args, model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int, int), rows, cols, channels
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    assert(mode in ("duel", "dqn"))
    with tf.compat.v1.variable_scope(model_name):
        input_data = Input(shape = input_shape, name = "input")
        policy_data = Input(shape = policy_shape, name = "policy")

        # MLP for policy.
        policy_input = TimeDistributed(Dense(16, activation='linear', name = 'policy_to_16'))(policy_data)
        # Observation
        print('>>>> Defining Recurrent Modules...')
        input_data_expanded = Reshape((input_shape[0], input_shape[1], input_shape[2], 1), input_shape = input_shape) (input_data)
        input_data_TimeDistributed = Permute((3, 1, 2, 4), input_shape=input_shape)(input_data_expanded) # (D, H, W, Batch)
        h1 = TimeDistributed(Convolution2D(32, (8, 8), strides = 4, activation = "relu", name = "conv1"), \
            input_shape=(args.num_frames, input_shape[0], input_shape[1], 1))(input_data_TimeDistributed)
        h2 = TimeDistributed(Convolution2D(64, (4, 4), strides = 2, activation = "relu", name = "conv2"))(h1)
        h3 = TimeDistributed(Convolution2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3"))(h2)
        flatten_hidden = TimeDistributed(Flatten())(h3)
        hidden_input = TimeDistributed(Dense(256, activation = 'relu', name = 'flat_to_512')) (flatten_hidden)
        hidden_policy = TimeDistributed(Dense(256, activation = 'relu', name = 'flat_to_512_2')) (policy_input)
        hidden_total = Concatenate(axis=2)([hidden_input, hidden_policy])
        context = LSTM(256, return_sequences=False, stateful=False, input_shape=(args.num_frames, 512)) (hidden_total)
        
        if mode == "dqn":
            h4 = Dense(512, activation='relu', name = "fc")(context)
            output = Dense(num_actions, name = "output")(h4)
        elif mode == "duel":
            value_hidden = Dense(256, activation = 'relu', name = 'value_fc')(context)
            value = Dense(1, name = "value")(value_hidden)
            action_hidden = Dense(256, activation = 'relu', name = 'action_fc')(context)
            action = Dense(num_actions, name = "action")(action_hidden)
            action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
            output = Lambda(lambda x: x[0] + x[1] - x[2], name = 'output')([action, value, action_mean])
    
    model = Model(inputs = [input_data, policy_data], outputs = output)
    print(model.summary())
    return model

def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.compat.v1.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    with writer.as_default(step):
        tf.summary.scalar(summary_value.tag, summary_value.simple_value)
        # writer.add_summary(summary, step).eval()
    # tf.summary.create_file_writer

class IDQNAgent:
    """Class implementing IDQN.

    This is a basic outline of the functions/parameters to implement the DQNAgnet. 

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self, args, num_actions):
        self.num_actions = num_actions
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        policy_shape = (args.num_policies, 2)
        self.history_processor = [HistoryPreprocessor(args.num_frames - 1, args.num_policies-1) for _ in range(2)]
        self.atari_processor = AtariPreprocessor()
        self.memory = [ReplayMemory(args, self.num_actions) for _ in range(2)]
        self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon, args.exploration_steps)
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        self.num_burn_in = args.num_burn_in
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_frames = args.num_frames
        self.output_path = args.output
        self.output_path_videos = args.output + '/videos/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.net_mode

        self.q_network_1 = create_model(input_shape, policy_shape, num_actions, self.net_mode, args, "QNet_1")
        self.q_network_2 = create_model(input_shape, policy_shape, num_actions, self.net_mode, args, "QNet_2")
        self.target_network_1 = create_model(input_shape, policy_shape, num_actions, self.net_mode, args, "TargetNet_1")
        self.target_network_2 = create_model(input_shape, policy_shape, num_actions, self.net_mode, args, "TargetNet_2")
        print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
        self.eval_freq = args.eval_freq
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        # initialize target network
        self.target_network_1.set_weights(self.q_network_1.get_weights())
        self.target_network_2.set_weights(self.q_network_2.get_weights())
        self.q_network_1.compile()
        self.q_network_2.compile()
        self.target_network_1.compile()
        self.target_network_2.compile()
        self.final_model = None

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = RMSprop(lr = self.learning_rate, clipnorm = 5)
        self.writer = tf.summary.create_file_writer(logdir = self.output_path)


        print("*******__init__", input_shape)

    # Not required
    def compile(self, optimizer = None, loss_func = None):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is the place to create the target network, setup 
        loss function and any placeholders.
        """
        if loss_func is None:
            loss_func = mean_huber_loss
            # loss_func = 'mse'
        if optimizer is None:
            optimizer = Adam(lr = self.learning_rate)
            # optimizer = RMSprop(lr=0.00025)
        with tf.compat.v1.variable_scope("Loss"):
            state = Input(shape = (self.frame_height, self.frame_width, self.num_frames) , name = "states")
            policy = Input(shape = (self.num_policies, self.num_actions,) , name = "policies")
            action_mask = Input(shape = (self.num_actions,), name = "actions")
            qa_value = self.q_network([state, policy])
            qa_value = Multiply(name = "multiply")([qa_value, action_mask])
            qa_value = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims = True), name = "sum")(qa_value)

        self.final_model = Model(inputs = [state, policy, action_mask], outputs = qa_value)
        self.final_model.compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state, q_value):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # state = state[None, :, :, :]
        policy = np.random.normal(0.0, 1.0,(1, self.num_frames, self.num_actions))
        return self.q_network_1.predict_on_batch([np.expand_dims(state[0], axis = 0), np.expand_dims(q_value[0], axis = 0)]), \
        self.q_network_2.predict_on_batch([np.expand_dims(state[1], axis = 0), np.expand_dims(q_value[1], axis = 0)])

    def select_action(self, state, q_value, is_training = True, **kwargs):
        """Select the action based on the current state.

        Returns
        --------
        selected action
        """
        q_values_1, q_values_2 = self.calc_q_values(state, q_value)
        if is_training:
            if kwargs['policy_type'] == 'UniformRandomPolicy':
                return [UniformRandomPolicy(self.num_actions).select_action(), \
                UniformRandomPolicy(self.num_actions).select_action()], \
                [q_values_1, q_values_2]
            else:
                # linear decay greedy epsilon policy
                return [self.policy.select_action(q_values_1, is_training), \
                self.policy.select_action(q_values_2, is_training)], \
                [q_values_1, q_values_2]
        else:
            return [GreedyEpsilonPolicy(0.05).select_action(q_values_1), \
            GreedyEpsilonPolicy(0.05).select_action(q_values_2)], \
            [q_values_1, q_values_2]

    def update_policy(self, current_sample = None, agent = None):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        batch_size = self.batch_size
        
        if self.no_experience:
            states = np.stack([current_sample.state])
            next_states = np.stack([current_sample.next_state])
            rewards = np.asarray([current_sample.reward])
            mask = np.asarray([1 - int(current_sample.is_terminal)])

            action_mask = np.zeros((1, self.num_actions))
            action_mask[0, current_sample.action] = 1.0
        else:
            samples = [self.memory[i].sample(batch_size) for i in range(2)]
            assert len(samples) == 2
            samples = [self.atari_processor.process_batch(sample) for sample in samples]
            
            states_1 = np.stack([x.state for x in samples[0]])
            states_2 = np.stack([x.state for x in samples[1]])

            actions_1 = np.asarray([x.action for x in samples[0]])
            actions_2 = np.asarray([x.action for x in samples[1]])
            
            action_mask_1 = np.zeros((batch_size, self.num_actions))
            action_mask_2 = np.zeros((batch_size, self.num_actions))
            
            action_mask_1[range(batch_size), actions_1] = 1.0
            action_mask_2[range(batch_size), actions_2] = 1.0

            next_states_1 = np.stack([x.next_state for x in samples[0]])
            next_states_2 = np.stack([x.next_state for x in samples[1]])
            
            mask_1 = np.asarray([1 - int(x.is_terminal) for x in samples[0]])
            mask_2 = np.asarray([1 - int(x.is_terminal) for x in samples[1]])
            
            rewards_1 = np.asarray([x.reward for x in samples[0]])
            rewards_2 = np.asarray([x.reward for x in samples[1]])
            
            policy_1 = np.asarray([x.q_value for x in samples[0]])
            policy_2 = np.asarray([x.q_value for x in samples[1]])

            next_policy_1 = np.asarray([x.next_q_value for x in samples[0]])
            next_policy_2 = np.asarray([x.next_q_value for x in samples[1]])

        if self.no_target:
            next_qa_value_1 = self.q_network_1.predict_on_batch([next_states_1, next_policy_1])
            next_qa_value_2 = self.q_network_2.predict_on_batch([next_states_2, next_policy_2])
        else:
            next_qa_value_1 = self.target_network_1.predict_on_batch([next_states_1, next_policy_1])
            next_qa_value_2 = self.target_network_2.predict_on_batch([next_states_2, next_policy_2])

        if self.enable_ddqn:
            qa_value_1 = self.q_network_1.predict_on_batch([next_states_1, next_policy_1])
            qa_value_2 = self.q_network_2.predict_on_batch([next_states_2, next_policy_2])
            max_actions_1 = np.argmax(qa_value_1, axis = 1)
            max_actions_2 = np.argmax(qa_value_2, axis = 1)
            next_qa_value_1 = next_qa_value_1[range(batch_size), max_actions_1]
            next_qa_value_2 = next_qa_value_2[range(batch_size), max_actions_2]
        else:
            next_qa_value_1 = np.max(next_qa_value_1, axis = 1)
            next_qa_value_2 = np.max(next_qa_value_2, axis = 1)
        target_1 = rewards_1 + self.gamma * mask_1 * next_qa_value_1
        target_2 = rewards_2 + self.gamma * mask_2 * next_qa_value_2
        
        with tf.GradientTape() as tape:
            qa_value_1 = self.q_network_1([states_1, policy_1])
            qa_value_1 = Multiply(name = "multiply_1")([qa_value_1, action_mask_1])
            qa_value_1 = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims = True), name = "sum_1")(qa_value_1)
            loss_value_1 = self.loss_fn(target_1, tf.squeeze(qa_value_1, axis= 1))
        grads = tape.gradient(loss_value_1, self.q_network_1.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q_network_1.trainable_weights))
        
        with tf.GradientTape() as tape:
            qa_value_2 = self.q_network_2([states_2, policy_2])
            qa_value_2 = Multiply(name = "multiply_2")([qa_value_2, action_mask_2])
            qa_value_2 = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims = True), name = "sum_1")(qa_value_2)
            loss_value_2 = self.loss_fn(target_2, tf.squeeze(qa_value_2, axis= 1))
        grads = tape.gradient(loss_value_2, self.q_network_2.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q_network_2.trainable_weights))
        
        return loss_value_1.numpy().mean() + loss_value_2.numpy().mean(), \
        np.mean(target_1) + np.mean(target_2)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        This is where you sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is the Atari environment. 
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        is_training = True
        print("Training starts.")
        self.save_model(0)
        eval_count = 0

        state = env.reset()
        burn_in = True
        idx_episode = 1
        episode_loss = .0
        episode_frames = 0
        episode_reward = .0
        episode_raw_reward = .0
        episode_target_value = .0

        # Logs
        losses_list = list()
        step_loss_list = list()
        step_reward = 0.0
        step_reward_raw = 0.0
        q_value = [np.random.normal(0.0, 1.0,(1, self.num_actions)) for _ in range(2)]


        for t in range(self.num_burn_in + num_iterations):
            action_state = [self.history_processor[i].process_state_for_network(
                self.atari_processor.process_state_for_network(s)) for i, s in enumerate(state)]

            action_policy = [self.history_processor[i].process_policy_for_network(s) for i, s in enumerate(q_value)]
            policy_type = "UniformRandomPolicy" if burn_in else "LinearDecayGreedyEpsilonPolicy"
            action, q_value = self.select_action(action_state, action_policy[::-1], is_training, policy_type = policy_type)
            processed_state = [self.atari_processor.process_state_for_memory(s) for s in state]

            state, reward, done, info = env.step(action)

            processed_next_state = [self.atari_processor.process_state_for_network(s) for s in state]

            processed_reward = self.atari_processor.process_reward(reward)
            # append the qvalue of the other agent.
            for i in range(2):
                self.memory[i].append(processed_state[i], action[i], processed_reward, done, q_value[-(i+1)])
           
            if not burn_in: 
                episode_frames += 1
                episode_reward += processed_reward
                episode_raw_reward += reward
                if episode_frames > max_episode_length:
                    done = True

            if not burn_in:
                step_reward += processed_reward
                step_reward_raw += reward
                step_losses = [t-last_burn-1, step_reward, step_reward_raw, step_reward / (t-last_burn-1), step_reward_raw / (t-last_burn-1)]
                step_loss_list.append(step_losses)


            if done:
                if not burn_in:
                    avg_target_value = episode_target_value / episode_frames
                    print(">>> Training: time %d, episode %d, length %d, reward %.0f, raw_reward %.0f, loss %.4f, target value %.4f, policy step %d, memory cap %d" % 
                        (t, idx_episode, episode_frames, episode_reward, episode_raw_reward, episode_loss, 
                        avg_target_value, self.policy.step, self.memory[0].current))
                    sys.stdout.flush()
                    save_scalar(idx_episode, 'train/episode_frames', episode_frames, self.writer)
                    save_scalar(idx_episode, 'train/episode_reward', episode_reward, self.writer)
                    save_scalar(idx_episode, 'train/episode_raw_reward', episode_raw_reward, self.writer)
                    save_scalar(idx_episode, 'train/episode_loss', episode_loss, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_reward', episode_reward / episode_frames, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_target_value', avg_target_value, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_loss', episode_loss / episode_frames, self.writer)

                    # log losses
                    losses = [idx_episode, episode_frames, episode_reward, episode_raw_reward, episode_loss, episode_reward / episode_frames, avg_target_value, episode_loss / episode_frames]
                    losses_list.append(losses)

                    # reset values
                    episode_frames = 0
                    episode_reward = .0
                    episode_raw_reward = .0
                    episode_loss = .0
                    episode_target_value = .0
                    idx_episode += 1
                burn_in = (t < self.num_burn_in)
                state = env.reset()
                self.atari_processor.reset()
                for i in range(2):
                    self.history_processor[i].reset()

            if burn_in:
                last_burn = t

            if not burn_in:
                if t % self.train_freq == 0:
                    loss, target_value = self.update_policy()
                    episode_loss += loss
                    episode_target_value += target_value
                # update freq is based on train_freq
                if t % (self.train_freq * self.target_update_freq) == 0:
                    # target updates can have the option to be hard or soft
                    # related functions are defined in deeprl_prj.utils
                    # here we use hard target update as default
                    self.target_network_1.set_weights(self.q_network_1.get_weights())
                    self.target_network_2.set_weights(self.q_network_2.get_weights())
                if t % self.save_freq == 0:
                    self.save_model(idx_episode)

                    loss_array = np.asarray(losses_list)
                    print (loss_array.shape) # 10 element vector

                    loss_path = self.output_path + "/losses/loss_episodes" + str(idx_episode) + ".csv"
                    np.savetxt(loss_path, loss_array, fmt='%.5f', delimiter=',')

                    step_loss_array = np.asarray(step_loss_list)
                    print (step_loss_array.shape) # 10 element vector

                    step_loss_path = self.output_path + "/losses/loss_steps" + str(t-last_burn-1) + ".csv"
                    np.savetxt(step_loss_path, step_loss_array, fmt='%.5f', delimiter=',')

        self.save_model(idx_episode)


    def save_model(self, idx_episode):
        # pass
        # for i in range(2):
        safe_path = self.output_path + "/qnet_1_" + str(idx_episode) + ".h5"
        self.q_network_1.save_weights(safe_path)
        safe_path = self.output_path + "/qnet_2_" + str(idx_episode) + ".h5"
        self.q_network_2.save_weights(safe_path)
        print("Network at", idx_episode, "saved to:", safe_path)

 