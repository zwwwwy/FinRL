from typing import List, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from collections.abc import Generator

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize
from torch import _tensor


class DapoBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    # old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DapoBuffer(BaseBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray  # advantages + value
    episode_starts: np.ndarray
    log_probs: np.ndarray
    # values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        优势函数的计算方法改变，不需要gae的lambda和gamma
        rewards应存储调整后reward
        gamma 和 gae_lambda 仅用于兼容 OnPolicyAlgorithm 的构造接口，不参与计算。
        """
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.group_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        # value: th.Tensor,
        log_prob: th.Tensor,
        group_id: np.ndarray,
    ):
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        # self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.group_ids[self.pos] = np.asarray(group_id, dtype=np.int32)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get_advantages(self) -> List[np.ndarray]:
        """
        buffer已经在get验证已满

        masks为每个环境中未被丢弃的数据
        """
        # keep_group_id = [[] for _ in range(self.n_envs)]  # 存储每个环境中保留的id
        masks = [np.zeros(self.buffer_size, dtype=bool) for _ in range(self.n_envs)]
        for env in range(self.n_envs):
            unique_group_ids = np.unique(self.group_ids[:, env])
            for current_group_id in unique_group_ids:
                indices = np.where(self.group_ids[:, env] == current_group_id)[0]
                rewards = self.rewards[indices, env]
                std_t = np.std(rewards)
                if std_t < 1e-6:
                    continue
                mean_t = np.mean(rewards)
                advantages = (rewards - mean_t) / (std_t + 1e-8)

                # keep_group_id[env].append(current_group_id)
                self.advantages[indices, env] = advantages
                masks[env][indices] = True

        return masks

    def get(self, batch_size: int | None = None) -> Generator[DapoBufferSamples, None, None]:
        assert self.full, ""
        masks = self.get_advantages()
        mask = np.concatenate(masks, axis=0).astype(bool)
        remain_space = np.arange(self.buffer_size * self.n_envs)[mask]  # 剩余的group

        indices = np.random.permutation(remain_space)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                # "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < len(indices):
            batch_inds = indices[start_idx: start_idx + batch_size]
            yield self._get_samples(batch_inds)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DapoBufferSamples:
        data = (
            self.observations[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions[batch_inds].astype(np.float32, copy=False),
            # self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return DapoBufferSamples(*tuple(map(self.to_torch, data)))
