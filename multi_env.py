"""
Based on OpenAI's baselines repo
"""

from collections import OrderedDict

import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.tile_images import tile_images
from gym import spaces


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


def make_buffered_env_wrapper(env_factory):
    def wrapped():
        return _BufferedMultiEnv(env_factory)

    return wrapped


class MultiEnvWrapper:
    def __init__(self, env_factories, num_workers=1):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_factories)
        self.num_workers = min(num_workers, self.num_envs)

        if self.num_workers > 1:
            self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_workers)])
            # distribute envs over workers
            env_dispatch = np.array_split(env_factories, self.num_workers)
            buffered_envs = [make_buffered_env_wrapper(envs) for envs in env_dispatch]

            self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                       for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, buffered_envs)]
            for p in self.ps:
                p.daemon = True  # if the main process crashes, we should not cause things to hang
                p.start()
            for remote in self.work_remotes:
                remote.close()

            self.remotes[0].send(('get_spaces', None))
            self.observation_space, self.action_space = self.remotes[0].recv()

        else:
            self.envs = _BufferedMultiEnv(env_factories)
            self.observation_space = self.envs.observation_space
            self.action_space = self.envs.action_space
            self._actions = None

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        if self.num_workers <= 1:
            self._actions = actions
            self.waiting = True
            return
        for remote, action in zip(self.remotes, np.array_split(actions, self.num_workers)):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        if self.num_workers <= 1:
            envs_step = self.envs.step(self._actions)
            self.waiting = False
            self._actions = None
            return envs_step

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.concatenate(obs), np.concatenate(rews), np.concatenate(dones), infos

    def reset(self):
        if self.num_workers <= 1:
            return self.envs.reset()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.concatenate([remote.recv() for remote in self.remotes])

    def close(self):
        if self.num_workers <= 1:
            return self.envs.close()
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        if self.num_workers <= 1:
            return self.envs.render(mode=mode)
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = np.asarray([pipe.recv() for pipe in self.remotes])
        imgs = imgs.reshape((-1, *imgs.shape[2:]))
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError


class _BufferedMultiEnv:
    def __init__(self, env_fatories):
        self.num_envs = len(env_fatories)
        self.envs = [fn() for fn in env_fatories]
        env = self.envs[0]

        shapes, dtypes = {}, {}
        self.keys = []
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        if isinstance(self.observation_space, spaces.Dict):
            assert isinstance(self.observation_space.spaces, OrderedDict)
            subspaces = self.observation_space.spaces
        else:
            subspaces = {None: self.observation_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step(self, actions):
        for e in range(self.num_envs):
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(actions[e])
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def render(self, mode='human'):
        return [e.render(mode=mode) for e in self.envs]

    def close(self):
        for e in self.envs:
            e.close()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
