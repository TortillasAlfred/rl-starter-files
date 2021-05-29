import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", required=True, help="name of the trained model (REQUIRED)"
)
parser.add_argument(
    "--episodes",
    type=int,
    default=1000,
    help="number of episodes of evaluation (default: 100)",
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--procs", type=int, default=16, help="number of processes (default: 16)"
)
parser.add_argument(
    "--argmax",
    action="store_true",
    default=False,
    help="action with highest probability is selected",
)
parser.add_argument(
    "--worst-episodes-to-show",
    type=int,
    default=10,
    help="how many worst episodes to show",
)
parser.add_argument(
    "--memory", action="store_true", default=False, help="add a LSTM to the model"
)
parser.add_argument(
    "--text", action="store_true", default=False, help="add a GRU to the model"
)
parser.add_argument(
    "--delta", type=float, default=0.05, help="env stochasticity (default: 0.05)"
)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    # Overrode the old flexible env loading to only load our stochastic env
    # envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    envs.append(utils.get_stochastic_env(seed=args.seed + 10000 * i, delta=args.delta))
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    device=device,
    argmax=args.argmax,
    num_envs=args.procs,
    use_memory=args.memory,
    use_text=args.text,
)
print("Agent loaded\n")

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)

while log_done_counter < args.episodes:
    actions = agent.get_actions(obss)
    obss, rewards, dones, _ = env.step(actions)
    agent.analyze_feedbacks(rewards, dones)

    log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

    mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames / (end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print(
    "F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}".format(
        num_frames,
        fps,
        duration,
        *return_per_episode.values(),
        *num_frames_per_episode.values(),
    )
)

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(
        range(len(logs["return_per_episode"])),
        key=lambda k: logs["return_per_episode"][k],
    )
    for i in indexes[:n]:
        print(
            "- episode {}: R={}, F={}".format(
                i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]
            )
        )

# NEW: Plot learned policy directly on the maze

# TODO: Plot learned policy with argmax and no possible perturbation
env = utils.get_stochastic_env(seed=args.seed, delta=0.0)

agent = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    device=device,
    argmax=True,
    num_envs=args.procs,
    use_memory=args.memory,
    use_text=args.text,
)
print("Agent reloaded\n")

all_obs = []

obs = env.reset()
all_obs.append(obs)

while not done:
    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    all_obs.append(obs)
    agent.analyze_feedback(reward, done)

# NEW: plot PDF and CDF of returns

# CDF
plot = sns.ecdfplot(data=logs["return_per_episode"])
plot.set(xlim=(0, 1), title="Returns CDF", xlabel="Percentile", ylabel="Return")
plt.savefig(join(model_dir, "cdf.png"))
plt.close()

# PDF
plot = sns.kdeplot(logs["return_per_episode"], fill=True, alpha=0.5)
plot.set(title="Return Distribution", xlabel="Return")
plt.savefig(join(model_dir, "pdf.png"))
plt.close()
