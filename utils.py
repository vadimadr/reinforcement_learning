import gym


def get_cumulative_rewards(rewards, gamma=0.99):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """

    rewards = list(reversed(rewards))
    cumulative_rewards = [rewards[0]]
    for r in rewards[1:]:
        cumulative_rewards.append(r + gamma * cumulative_rewards[-1])
    return list(reversed(cumulative_rewards))


def generate_session(agent, env, t_max=1000):
    """
    play a full session with provided agent and returns sequences of states, actions and rewards
    """
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()
    for t in range(t_max):
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


def make_fun(agent, env, record_video=False, render=True, n_episodes=1):
    if record_video:
        env = gym.wrappers.Monitor(env, directory='videos', force=True)

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
            action = agent.get_action(state)
            new_s, reward, done, _ = env.step(action)
            total_reward += reward
            state = new_s
        print("Episode %d total reward: %f" % (episode, total_reward))
    env.close()
