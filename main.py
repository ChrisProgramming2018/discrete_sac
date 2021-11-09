from Brain.agent import SAC
import time
from Common.play import Play
from Common.utils import *
from Common.logger import Logger
from Common.config import get_params


def intro_env():
    for e in range(5):
        test_env.reset()
        d = False
        ep_r = 0
        while not d:
            a = test_env.env.action_space.sample()
            _, r, d, info = test_env.step(a)
            ep_r += r
            test_env.env.render()
            time.sleep(0.005)
            print(f"reward: {np.sign(r)}")
            print(info)
            if d:
                break
        print("episode reward: ", ep_r)
    test_env.close()
    exit(0)


if __name__ == "__main__":
    params = get_params()

    test_env = gym.make(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})

    print(f"Number of actions: {params['n_actions']}")

    if params["do_intro_env"]:
        intro_env()

    env = gym.make(params["env_name"])
    print("state shape ", env.observation_space.shape[0])
    params["state_shape"] = env.observation_space.shape[0]
    print("action shape ", env.action_space.n)
    params["n_actions"] =  env.action_space.n
    agent = SAC(**params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode = logger.load_weights()
            agent.hard_update_target_network()
            agent.alpha = agent.log_alpha.exp()
            min_episode = episode
            print("Keep training from previous run.")

        else:
            min_episode = 0
            print("Train from scratch.")

        state = env.reset()
        episode_reward = 0
        alpha_loss, q_loss, policy_loss = 0, 0, 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, params["max_steps"] + 1):
            if step < params['initial_random_steps']:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                reward = np.sign(reward)
                agent.store(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
            else:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                if step % params["train_period"] == 0:
                    alpha_loss, q_loss, policy_loss = agent.train()

                if done:
                    logger.off()
                    logger.log(episode, episode_reward, alpha_loss, q_loss, policy_loss , step)

                    episode += 1
                    state = env.reset()
                    episode_reward = 0
                    episode_loss = 0
                    logger.on()

    logger.load_weights()
    player = Play(env, agent, params)
    player.evaluate()
