import gym
import numpy as np


def get_cumulative_rewards(rewards, gamma=0.99, dones=None):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    rewards = np.asarray(rewards)
    if dones is None:
        dones = np.zeros(rewards.shape[-1])
    dones = np.asarray(dones)

    batch_size = rewards.shape[0] if len(rewards.shape) > 1 else 1
    cumulative_rewards = []
    current_reward = np.zeros(batch_size)
    for r, done in zip(rewards.T[::-1], dones.T[::-1]):
        # do not discount if it is the last observation in a session
        current_reward = r + gamma * current_reward * (1 - done)
        cumulative_rewards.append(current_reward)

    return np.stack(cumulative_rewards[::-1], axis=1).reshape(rewards.shape)


def get_total_rewards(rewards, dones):
    """
    Calculate total reward for all sessions in a batch
    """
    rewards = np.asarray(rewards)
    dones = np.array(dones)
    dones = dones.reshape((-1, dones.shape[-1]))
    dones[:, -1] = False

    total_rewards = []
    current_total_reward = 0
    for reward, done in zip(rewards.flat, dones.flat):
        current_total_reward += reward
        if done:
            total_rewards.append(current_total_reward)
            current_total_reward = 0
    return total_rewards


def generate_session(agent, env, max_num_steps=1000):
    """
    play a full session with provided agent and returns sequences of states, actions and rewards
    """
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()
    for t in range(max_num_steps):
        # action probabilities array aka pi(a|s)
        a = agent.get_action(s)
        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    return states, actions, rewards


def generate_session_batch(agent, envs, num_steps=32):
    """
    Play session in multiple environments and generate a batch of (states, actions, rewards, dones)
    """
    # arrays to record session
    states, actions, rewards, dones = [], [], [], []
    s = envs.reset()
    for t in range(num_steps):
        # action probabilities array aka pi(a|s)
        a = agent.get_action(s)
        new_s, r, done, info = envs.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dones.append(done)

        s = new_s

    states = np.asarray(states, dtype=np.float32).swapaxes(0, 1)
    actions = np.asarray(actions, dtype=np.int32).swapaxes(0, 1)
    rewards = np.asarray(rewards, dtype=np.float32).swapaxes(0, 1)
    dones = np.asarray(dones, dtype=bool).swapaxes(0, 1)

    last_value = agent.get_value(s)

    return states, actions, rewards, dones, last_value


def make_fun(agent, env, record_video=False, render=True, n_episodes=1):
    if record_video:
        env = gym.wrappers.Monitor(env, directory='videos', force=True)

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not np.asarray(done).all():
            if render:
                env.render()
            action = agent.get_action(state)
            new_s, reward, done, _ = env.step(action)
            total_reward += reward
            state = new_s
        print("Episode %d total reward: %f" % (episode, total_reward))
    env.close()
