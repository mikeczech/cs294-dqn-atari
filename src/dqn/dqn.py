import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
import time
import tensorflow as tf
from collections import namedtuple
from dqn_utils import *


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

LOG_EVERY_N_STEPS = 10000

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    q = q_func(obs_t_float, num_actions, "q_func", reuse=False)

    target_q = q_func(obs_tp1_float, num_actions, "target_q_func", reuse=False)
    y = rew_t_ph + (1 - done_mask_ph) * gamma * tf.reduce_max(target_q, axis=1)

    q_val = tf.reduce_sum(tf.one_hot(act_t_ph, depth=num_actions) * q, axis=1)
    total_error = tf.reduce_sum(tf.squared_difference(q_val, y))

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()

    checkpoint_dir_name = time.time()
    saver = tf.train.Saver(max_to_keep=10)
    summary_writer = tf.summary.FileWriter(os.environ['TF_LOGDIR'], session.graph)
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        idx = replay_buffer.store_frame(last_obs)
        encoded_last_obs = replay_buffer.encode_recent_observation().reshape(1, *input_shape)

        # use an epsilon-decreasing strategy
        choose_random_action = np.random.binomial(1, exploration.value(t))

        if model_initialized and not choose_random_action:
            last_obs_q = session.run(q, { obs_t_ph: encoded_last_obs })
            action = np.argmax(last_obs_q[0])
        else:
            action = random.randint(0, num_actions - 1)

        obs, reward, done, info = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)

        if done:
            last_obs = env.reset()
        else:
            last_obs = obs

        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {
                    obs_t_ph: obs_batch,
                    obs_tp1_ph: next_obs_batch,
                })
                model_initialized = True

            session.run(train_fn, feed_dict={
                obs_t_ph: obs_batch,
                act_t_ph: act_batch,
                rew_t_ph: rew_batch,
                obs_tp1_ph: next_obs_batch,
                done_mask_ph: done_mask,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            })

            num_param_updates += 1
            if num_param_updates == target_update_freq:
                session.run(update_target_fn)
                saver.save(session, 'checkpoints/{}/model'.format(checkpoint_dir_name), global_step=t)
                num_param_updates = 0

        log_process(summary_writer, env, t, mean_episode_reward, best_mean_episode_reward, model_initialized, optimizer_spec, exploration)


def log_process(summary_writer, env, t, mean_episode_reward, best_mean_episode_reward, model_initialized, optimizer_spec, exploration):
    episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
    if len(episode_rewards) > 0:
        mean_episode_reward = np.mean(episode_rewards[-100:])
        reward_summary = tf.Summary()
        reward_summary.value.add(tag='mean_episode_reward', simple_value=mean_episode_reward)
        summary_writer.add_summary(reward_summary, t)

    if len(episode_rewards) > 100:
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

    if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
        print("Timestep %d" % (t,))
        print("mean reward (100 episodes) %f" % mean_episode_reward)
        print("best mean reward %f" % best_mean_episode_reward)
        print("episodes %d" % len(episode_rewards))
        print("exploration %f" % exploration.value(t))
        print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))

        summary_writer.flush()
        sys.stdout.flush()
