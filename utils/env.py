import gym
import gym_minigrid

from gym_minigrid.minigrid import *
from gym_minigrid.envs import DistShiftEnv
import numpy as np


class StochasticDistShiftEnv(DistShiftEnv):
    """
    Stochastic Distributional shift environment.
    """

    def __init__(
        self,
        width=9,
        height=15,
        agent_start_pos=np.array((1, 1)),
        agent_start_dir=0,
        strip2_row=2,
        delta=0.05,  # Random proba
    ):
        self.delta = delta

        super().__init__(
            width=width,
            height=height,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            strip2_row=strip2_row,
        )

    def _move_up(self):
        reward = 0
        done = False

        up_pos = self.agent_pos + DIR_TO_VEC[-1]

        # Cannot fall outside of limits
        if up_pos[-1] <= 0:
            up_pos = self.agent_pos

        up_cell = self.grid.get(*up_pos)

        if up_cell == None or up_cell.can_overlap():
            self.agent_pos = up_pos
        if up_cell != None and up_cell.type == "goal":
            done = True
            reward = self._reward()
        if up_cell != None and up_cell.type == "lava":
            done = True

        return reward, done

    def step(self, action):
        self.step_count += 1

        if self._rand_float(0, 1) < self.delta:
            reward, done = self._move_up()
        else:
            reward = 0
            done = False

            # Get the position in front of the agent
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4

            # Rotate right
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agent_pos = fwd_pos
                if fwd_cell != None and fwd_cell.type == "goal":
                    done = True
                    reward = self._reward()
                if fwd_cell != None and fwd_cell.type == "lava":
                    done = True

            # Pick up an object
            elif action == self.actions.pickup:
                if fwd_cell and fwd_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = fwd_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

            # Drop an object
            elif action == self.actions.drop:
                if not fwd_cell and self.carrying:
                    self.grid.set(*fwd_pos, self.carrying)
                    self.carrying.cur_pos = fwd_pos
                    self.carrying = None

            # Toggle/activate an object
            elif action == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Done action (not used by default)
            elif action == self.actions.done:
                pass

            else:
                assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

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
        }

        return obs


class StochasticDistShift1(StochasticDistShiftEnv):
    def __init__(self, **kwargs):
        super().__init__(strip2_row=2, **kwargs)


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env


def get_stochastic_env(seed=None, **kwargs):
    env = StochasticDistShift1(**kwargs)
    env.seed(seed)
    return env
