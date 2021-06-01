from utils.env import StochasticDistShiftEnv

import gym
import gym_minigrid

from gym_minigrid.minigrid import *
import numpy as np
import torch


class FixedAdversaryStochasticDistShift1(StochasticDistShiftEnv):
    """
    Stochastic Distributional shift environment with fixed adversary.
    """

    def __init__(
        self,
        adversary,
        adversary_budget,
        width=9,
        height=9,
        agent_start_pos=np.array((1, 1)),
        agent_start_dir=0,
        strip2_row=2,
        delta=0.05,  # Random proba
        eps=1e-6,
    ):
        self.adversary = adversary
        self.delta = delta
        self.adversary_budget = adversary_budget
        self.remaining_budget = self.adversary_budget
        self.eps = eps

        super().__init__(
            width=width,
            height=height,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            strip2_row=strip2_row,
        )

    def _fall_event(self):
        next_pos = self.agent_pos
        next_dir = self.agent_dir
        reward = 0
        done = False

        up_pos = self.agent_pos + DIR_TO_VEC[-1]

        # Cannot fall outside of limits
        if up_pos[-1] <= 0:
            up_pos = self.agent_pos

        up_cell = self.grid.get(*up_pos)

        if up_cell == None or up_cell.can_overlap():
            next_pos = up_pos
        if up_cell != None and up_cell.type == "goal":
            done = True
            reward = self._reward()
        if up_cell != None and up_cell.type == "lava":
            done = True

        return next_pos, next_dir, reward, done

    def _normal_event(self, action):
        next_pos = self.agent_pos
        next_dir = self.agent_dir
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            next_dir -= 1
            if next_dir < 0:
                next_dir += 4

        # Rotate right
        elif action == self.actions.right:
            next_dir = (next_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                next_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == "goal":
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == "lava":
                done = True

        else:
            assert False, "unknown action"

        return (next_pos, next_dir, reward, done)

    def _get_transition_probas(self, action):
        fall_state = self._fall_event()
        normal_state = self._normal_event(action)

        if (fall_state[0] == normal_state[0]).all() and fall_state[1] == normal_state[
            1
        ]:
            return [normal_state], [1.0]
        else:
            return [fall_state, normal_state], [self.delta, 1.0 - self.delta]

    def _get_state_index(self, state):
        (x, y), direction, _, _ = state

        index = 0

        index += x * self.grid.height * 4
        index += y * 4
        index += direction

        return index

    @property
    def device(self):
        return self.adversary.device

    @property
    def state_size(self):
        return self.grid.width * self.grid.height * 4

    def _preprocess_transitions(self, states, probas):
        torch_probas = torch.zeros(
            (self.state_size + 1,), dtype=float, device=self.device
        )

        for state, proba in zip(states, probas):
            state_idx = self._get_state_index(state)
            torch_probas[state_idx] = proba

        torch_probas[-1] = self.remaining_budget

        return torch_probas

    def _postprocess_transitions(self, perturbed_transitions, adv_obs):
        perturbed_transitions = perturbed_transitions / perturbed_transitions.sum()
        perturbations = (
            perturbed_transitions / adv_obs["transition_probas"][:-1].cpu().numpy()
        )
        perturbations = np.nan_to_num(perturbations, nan=1)
        perturbations = perturbations[perturbed_transitions > 0].tolist()

        perturbed_transitions = perturbed_transitions[
            perturbed_transitions > 0
        ].tolist()

        perturbations_sum = sum(perturbed_transitions)
        for i in range(len(perturbed_transitions)):
            perturbed_transitions[i] = perturbed_transitions[i] / perturbations_sum

        return perturbed_transitions, perturbations

    def _get_adversarial_perturbation(self, states, probas):
        adv_obs = self.gen_adv_obs(states, probas)
        with torch.no_grad():
            perturbed_transitions = self.adversary.get_new_proba(adv_obs)

        # Ensure adversary budget is respected
        perturbations = (
            perturbed_transitions / adv_obs["transition_probas"][:-1].cpu().numpy()
        )
        perturbations = np.nan_to_num(perturbations, nan=1)
        max_perturbation = perturbations.max()

        if max_perturbation > self.remaining_budget:
            if self.remaining_budget > 1 + self.eps:
                perturbations = (
                    max_perturbation
                    - self.remaining_budget
                    + (self.remaining_budget - 1) * perturbations
                ) / (max_perturbation - 1)
                perturbed_transitions = (
                    adv_obs["transition_probas"][:-1].cpu().numpy() * perturbations
                )
            else:
                perturbed_transitions = adv_obs["transition_probas"][:-1].cpu().numpy()

        # Cast back into the list format
        perturbed_transitions, perturbations = self._postprocess_transitions(
            perturbed_transitions, adv_obs
        )

        return perturbed_transitions, perturbations

    def step(self, action):
        self.step_count += 1

        states, probas = self._get_transition_probas(action)

        perturbed_probas, perturbations = self._get_adversarial_perturbation(
            states, probas
        )

        sampled_idx = np.random.choice(len(perturbed_probas), p=perturbed_probas)
        next_pos, next_dir, reward, done = states[sampled_idx]
        perturbation_applied = perturbations[sampled_idx]
        self.remaining_budget = self.remaining_budget / perturbation_applied

        self.agent_pos = next_pos
        self.agent_dir = next_dir

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0
        self.remaining_budget = self.adversary_budget

        # Return first observation
        obs = self.gen_obs()

        return obs

    def gen_adv_obs(self, states, probas):
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        torch_probas = self._preprocess_transitions(states, probas)

        obs = {
            "image": image,
            "direction": self.agent_dir,
            "position": self.agent_pos,
            "mission": self.mission,
            "transition_probas": torch_probas,
            "remaining_budget": self.remaining_budget,
        }

        return obs

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(
            self, "mission"
        ), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "position": self.agent_pos,
            "mission": self.mission,
            "remaining_budget": self.remaining_budget,
        }

        return obs


class FixedPolicyStochasticDistShift1(StochasticDistShiftEnv):
    """
    Stochastic Distributional shift environment with fixed policy.
    """

    def __init__(
        self,
        policy,
        adversary_budget,
        width=9,
        height=9,
        agent_start_pos=np.array((1, 1)),
        agent_start_dir=0,
        strip2_row=2,
        delta=0.05,  # Random proba
        eps=1e-6,
    ):
        self.policy = policy
        self.delta = delta
        self.adversary_budget = adversary_budget
        self.remaining_budget = self.adversary_budget
        self.eps = eps

        super().__init__(
            width=width,
            height=height,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            strip2_row=strip2_row,
        )

    def _fall_event(self):
        next_pos = self.agent_pos
        next_dir = self.agent_dir
        reward = 0
        done = False

        up_pos = self.agent_pos + DIR_TO_VEC[-1]

        # Cannot fall outside of limits
        if up_pos[-1] <= 0:
            up_pos = self.agent_pos

        up_cell = self.grid.get(*up_pos)

        if up_cell == None or up_cell.can_overlap():
            next_pos = up_pos
        if up_cell != None and up_cell.type == "goal":
            done = True
            reward = self._reward()
        if up_cell != None and up_cell.type == "lava":
            done = True

        return next_pos, next_dir, reward, done

    def _normal_event(self, action):
        next_pos = self.agent_pos
        next_dir = self.agent_dir
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            next_dir -= 1
            if next_dir < 0:
                next_dir += 4

        # Rotate right
        elif action == self.actions.right:
            next_dir = (next_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                next_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == "goal":
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == "lava":
                done = True

        else:
            assert False, "unknown action"

        return (next_pos, next_dir, reward, done)

    def _get_transition_probas(self, action):
        fall_state = self._fall_event()
        normal_state = self._normal_event(action)

        if (fall_state[0] == normal_state[0]).all() and fall_state[1] == normal_state[
            1
        ]:
            return [normal_state], [1.0]
        else:
            return [fall_state, normal_state], [self.delta, 1.0 - self.delta]

    def _get_state_index(self, state):
        (x, y), direction, _, _ = state

        index = 0

        index += x * self.grid.height * 4
        index += y * 4
        index += direction

        return index

    @property
    def device(self):
        return self.policy.device

    @property
    def state_size(self):
        return self.grid.width * self.grid.height * 4

    def _preprocess_transitions(self, states, probas):
        torch_probas = torch.zeros(
            (self.state_size + 1,), dtype=float, device=self.device
        )

        for state, proba in zip(states, probas):
            state_idx = self._get_state_index(state)
            torch_probas[state_idx] = proba

        torch_probas[-1] = self.remaining_budget

        return torch_probas

    def _postprocess_transitions(self, perturbed_transitions):
        adv_obs = self.gen_adv_obs()

        perturbed_transitions = perturbed_transitions / perturbed_transitions.sum()
        perturbations = (
            perturbed_transitions / adv_obs["transition_probas"][:-1].cpu().numpy()
        )
        perturbations = np.nan_to_num(perturbations, nan=1)
        perturbations = perturbations[perturbed_transitions > 0].tolist()

        perturbed_transitions = perturbed_transitions[
            perturbed_transitions > 0
        ].tolist()

        perturbations_sum = sum(perturbed_transitions)
        for i in range(len(perturbed_transitions)):
            perturbed_transitions[i] = perturbed_transitions[i] / perturbations_sum

        return perturbed_transitions, perturbations

    def _apply_budget_constraint(self, perturbations):
        adv_obs = self.gen_adv_obs()
        old_probas = adv_obs["transition_probas"][:-1].cpu().numpy()

        perturbations[old_probas == 0] = 1
        max_perturbation = perturbations.max()

        if max_perturbation > self.remaining_budget and len(self.states) > 1:
            if self.remaining_budget > 1 + 1e-6:
                perturbations = (
                    max_perturbation
                    - self.remaining_budget
                    + (self.remaining_budget - 1) * perturbations
                ) / (max_perturbation - 1)
                new_probas = old_probas * perturbations
            else:
                new_probas = old_probas
        else:
            new_probas = old_probas * perturbations

        new_probas[old_probas == 0] = 0
        new_probas = new_probas / new_probas.sum()

        return new_probas

    def step(self, perturbations):
        self.step_count += 1

        perturbed_probas = self._apply_budget_constraint(perturbations)
        perturbed_probas, perturbations = self._postprocess_transitions(
            perturbed_probas
        )

        sampled_idx = np.random.choice(len(perturbed_probas), p=perturbed_probas)
        next_pos, next_dir, reward, done = self.states[sampled_idx]
        perturbation_applied = perturbations[sampled_idx]
        self.remaining_budget = self.remaining_budget / perturbation_applied

        self.agent_pos = next_pos
        self.agent_dir = next_dir

        if self.step_count >= self.max_steps:
            done = True

        action = self.policy.get_action(self.gen_obs())
        self.states, self.probas = self._get_transition_probas(action)

        obs = self.gen_adv_obs()

        return obs, -reward, done, {}

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0
        self.remaining_budget = self.adversary_budget

        # Return first observation
        action = self.policy.get_action(self.gen_obs())
        self.states, self.probas = self._get_transition_probas(action)

        obs = self.gen_adv_obs()

        return obs

    def gen_adv_obs(self):
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        torch_probas = self._preprocess_transitions(self.states, self.probas)

        obs = {
            "image": image,
            "direction": self.agent_dir,
            "position": self.agent_pos,
            "mission": self.mission,
            "transition_probas": torch_probas,
            "remaining_budget": self.remaining_budget,
        }

        return obs

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(
            self, "mission"
        ), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "position": self.agent_pos,
            "mission": self.mission,
            "remaining_budget": self.remaining_budget,
        }

        return obs


def get_stochastic_fixed_adversary_env(
    adversary, adversary_budget, seed=None, **kwargs
):
    env = FixedAdversaryStochasticDistShift1(adversary, adversary_budget, **kwargs)
    env.seed(seed)
    return env


def get_stochastic_fixed_policy_env(policy, adversary_budget, seed=None, **kwargs):
    env = FixedPolicyStochasticDistShift1(policy, adversary_budget, **kwargs)
    env.seed(seed)
    return env
