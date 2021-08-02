import gym
from multiprocessing import Process, Pipe

from gym_minigrid.minigrid import *
from gym_minigrid.envs import DistShiftEnv
import numpy as np
import torch


class StochasticDistShiftEnv(DistShiftEnv):
    """
    Stochastic Distributional shift environment.
    """

    class RestrictedActions(IntEnum):
        # Turn left, turn right, move forward
        up = 0
        down = 1
        right = 2
        left = 3

    ACTION_DIR_VEC = {
        RestrictedActions.up: np.array([0, -1]),
        RestrictedActions.down: np.array([0, 1]),
        RestrictedActions.right: np.array([1, 0]),
        RestrictedActions.left: np.array([-1, 0]),
    }

    def __init__(
        self,
        width=12,
        height=9,
        agent_start_pos=np.array((1, 1)),
        agent_start_dir=0,
        strip2_row=2,
        delta=0.05,  # Random proba
        adversary_budget=1.0,
        device="cpu",
    ):
        self.delta = delta
        self.adversary_budget = adversary_budget
        self.remaining_budget = self.adversary_budget
        self.device = device

        super().__init__(
            width=width,
            height=height,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            strip2_row=strip2_row,
        )

        # This is not a conventionnal Gym Env, don't want any unwanted side-effects
        del self.observation_space
        del self.action_space
        del self.agent_dir

        # Define obs and action spaces for both the adversary and policy players.
        self.adv_action_space = gym.spaces.Box(low=0.0, high=self.adversary_budget, shape=(4,), dtype=np.float32)
        self.adv_observation_space = gym.spaces.Dict(
            dict(
                pos=gym.spaces.Box(np.array([1, 1]), np.array([width - 2, height - 2])),
                probas=gym.spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            )
        )

        self.pol_actions = StochasticDistShiftEnv.RestrictedActions
        self.pol_action_space = gym.spaces.Discrete(len(self.pol_actions))
        self.pol_observation_space = gym.spaces.Box(np.array([1, 1]), np.array([width - 2, height - 2]))

    def step(self, action):
        raise NotImplementedError(
            "This environment does not support step(). Use the step_policy() and step_adversary() functions instead."
        )

    def _is_oob(self, position):
        return position[0] < 1 or position[1] < 1 or position[0] > self.width - 2 or position[1] > self.height - 2

    def _hypothetical_transition(self, policy_action):
        reward = -1
        done = False

        next_pos = self.agent_pos + self.ACTION_DIR_VEC[policy_action]
        if self._is_oob(next_pos):
            next_pos = self.agent_pos

        next_cell = self.grid.get(*next_pos)

        if next_cell != None and next_cell.type == "goal":
            done = True
            reward = 20
        if next_cell != None and next_cell.type == "lava":
            done = True

        reward = reward / 20

        return next_pos, reward, done

    def _get_transitions(self, policy_action):
        """
        Method called in step_policy() to get the possible next states, rewards, dones and their probas.

        Returns lists respectively containing a transition proba, its next state, its reward
        and if it is a terminal state.

        The stochasticity is defined as the agent following its selected action with probability 1 - delta
        and taking any of the 3 other possible actions with proba delta/3.
        """
        chosen_proba = 1 - self.delta
        random_proba = (
            self.delta / 3
        )  # Since we have 4 actions, each non-selected action has p=delta/3 of being sampled

        transition_probas = [random_proba] * 4
        transition_probas[policy_action] = chosen_proba

        states, rewards, dones = [] * 4, [] * 4, [] * 4
        for possible_action in self.pol_action_space:
            s, r, d = self._hypothetical_transition(possible_action)
            states[possible_action] = s
            rewards[possible_action] = r
            dones[possible_action] = d

        return transition_probas, states, rewards, dones

    def step_policy(self, policy_action):
        """
        policy_action: integer between 0 and 3 to characterize respectively (up, down, right, left).
        """
        self.probas, self.states, self.rewards, self.dones = self._get_transitions(policy_action)

        obs = self.gen_adv_obs()

        return obs, None, None, {}

    def _make_transition(self, adversary_perturbation):
        """
        TODO:
        1. Ensure adversary_perturbation is not over budget and generates a distribution
           (i.e. adversary_perturbation * self.probas sums up to 1)
        2. Sample the next state and reward from the perturbed transition probas. Return them.
        """

        return None

    def step_adversary(self, adversary_perturbation):
        """
        adversary_perturbation: Numpy vector of shape (4,). Indicates the adversary's moves
        and is therefore constrained to (1) generate a distribution over next states and
        (2) be within the remaining budget.
        """
        self.step_count += 1

        self.agent_pos, reward, done = self._make_transition(adversary_perturbation)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_pol_obs()

        return obs, reward, done, {}

    @property
    def torch_adv_state(self):
        torch_state = torch.tensor(self.agent_pos, dtype=float, device=self.device)
        torch_probas = torch.tensor(self.probas, dtype=float, device=self.device)
        torch_budget = torch.tensor([self.remaining_budget], dtype=float, device=self.device)

        return torch.stack(torch_state, torch_probas, torch_budget)

    def gen_adv_obs(self):
        obs = {
            "state": self.torch_adv_state,
            "remaining_budget": self.remaining_budget,
        }

        return obs

    def gen_pol_obs(self):
        obs = {
            "position": self.agent_pos,
        }

        return obs

    def reset(self):
        raise NotImplementedError(
            "This environment does not support reset(). Use the reset_policy() and reset_adversary() functions instead."
        )

    def reset_policy(self):
        """
        TODO:

        1. Reset all trajectory-related infos like agent_pos, step_count, etc.
        2. Return the same observation as the step_policy() function.
        """

        return None

    def reset_adversary(self):
        """
        TODO:

        1. Reset all trajectory-related infos like agent_pos, step_count, etc.
        2. Return the same observation as the step_adversary() function. For this
           case, the adversary can only act over the initial state distribution.
           In the minigrid example this is determinstic so this should be a one-hot
           distribution but for stochastic start envs like CartPole, this is another place
           where the adversary might try to apply perturbations.
        """

        return None


class StochasticDistShift1(StochasticDistShiftEnv):
    def __init__(self, **kwargs):
        super().__init__(strip2_row=2, **kwargs)


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step_policy":
            obs, reward, done, info = env.step_policy(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "step_adversary":
            obs, reward, done, info = env.step_adversary(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset_policy":
            obs = env.reset_policy()
            conn.send(obs)
        elif cmd == "reset_adversary":
            obs = env.reset_adversary()
            conn.send(obs)
        else:
            raise NotImplementedError


class ParallelAdvEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.pol_observation_space = self.envs[0].pol_observation_space
        self.pol_action_space = self.envs[0].pol_action_space
        self.adv_observation_space = self.envs[0].adv_observation_space
        self.adv_action_space = self.envs[0].adv_action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        raise NotImplementedError(
            "This environment does not support reset(). Use the reset_policy() and reset_adversary() functions instead."
        )

    def reset_policy(self):
        for local in self.locals:
            local.send(("reset_policy", None))
        results = [self.envs[0].reset_policy()] + [local.recv() for local in self.locals]
        return results

    def reset_adversary(self):
        for local in self.locals:
            local.send(("reset_adversary", None))
        results = [self.envs[0].reset_adversary()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        raise NotImplementedError(
            "This environment does not support step(). Use the step_policy() and step_adversary() functions instead."
        )

    def step_policy(self, policy_actions):
        for local, policy_action in zip(self.locals, policy_actions[1:]):
            local.send(("step_policy", policy_action))
        obs, reward, done, info = self.envs[0].step_policy(policy_actions[0])
        if done:
            obs = self.envs[0].reset_policy()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def step_adversary(self, adversary_actions):
        for local, adversary_action in zip(self.locals, adversary_actions[1:]):
            local.send(("step_adversary", adversary_action))
        obs, reward, done, info = self.envs[0].step_adversary(adversary_actions[0])
        if done:
            obs = self.envs[0].reset_adversary()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env


def get_stochastic_env(seed=None, **kwargs):
    env = StochasticDistShift1(**kwargs)
    env.seed(seed)
    return env
