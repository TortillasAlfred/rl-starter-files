import argparse
import time
import numpy
import torch

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", required=True, help="name of the trained model (REQUIRED)"
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--shift",
    type=int,
    default=0,
    help="number of times the environment is reset at the beginning (default: 0)",
)
parser.add_argument(
    "--argmax",
    action="store_true",
    default=False,
    help="select the action with highest probability (default: False)",
)
parser.add_argument(
    "--pause",
    type=float,
    default=0.1,
    help="pause duration between two consequent actions of the agent (default: 0.1)",
)
parser.add_argument(
    "--gif", type=str, default=None, help="store output as gif with the given filename"
)
parser.add_argument(
    "--episodes", type=int, default=1000000, help="number of episodes to visualize"
)
parser.add_argument(
    "--memory", action="store_true", default=False, help="add a LSTM to the model"
)
parser.add_argument(
    "--text", action="store_true", default=False, help="add a GRU to the model"
)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.get_stochastic_env()
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
policy_acmodel = utils.ACModel(env.observation_space, env.action_space)
agent = utils.Agent(
    policy_acmodel,
    env.observation_space,
    model_dir,
    device=device,
    argmax=True,
    num_envs=1,
    pretrained=True,
)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render("human")

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render("human", highlight=False)
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(-1)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
    print("Done.")
