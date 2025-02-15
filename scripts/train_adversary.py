import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys

import utils
from model import ACModel, AdversaryACModel
import numpy as np
import os
import shutil


def train(args):
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"adversary_training"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    dummy_env = utils.get_stochastic_env()

    policy_acmodel = ACModel(
        dummy_env.observation_space, dummy_env.action_space, args.mem, args.text
    )

    adversary_acmodel = AdversaryACModel(dummy_env)

    adversary_agent = utils.AdversaryAgent(
        adversary_acmodel,
        dummy_env,
        model_dir,
        device=device,
        num_envs=args.procs,
        pretrained=False,
    )

    policy_agent = utils.Agent(
        policy_acmodel,
        dummy_env.observation_space,
        utils.get_model_dir("policy_training"),
        argmax=True,
        device=device,
        num_envs=args.procs,
        pretrained=True,
    )

    adv_envs = []
    for i in range(args.procs):
        # Overrode the old flexible env loading to only load our stochastic env
        adv_envs.append(
            utils.get_stochastic_fixed_policy_env(
                policy_agent,
                1.0 / args.target_quantile,
                seed=args.seed + 10000 * i,
                delta=args.delta,
            )
        )
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    _, adv_preprocess_obss = utils.get_adversary_obss_preprocessor(
        adv_envs[0].observation_space
    )

    txt_logger.info("Observations preprocessor loaded")

    # Load model
    policy_acmodel.to(device)
    txt_logger.info("Policy model loaded\n")
    txt_logger.info("{}\n".format(policy_acmodel))

    adversary_acmodel.to(device)
    txt_logger.info("Adversary model loaded\n")
    txt_logger.info("{}\n".format(adversary_acmodel))

    # Load algo

    adv_algo = utils.ContinuousPPOAlgo(
        adv_envs,
        adversary_acmodel,
        device,
        args.frames_per_proc,
        args.discount,
        args.lr,
        args.lr_decay_rate,
        args.gae_lambda,
        args.entropy_coef,
        args.value_loss_coef,
        args.max_grad_norm,
        args.recurrence,
        args.optim_eps,
        args.clip_eps,
        args.epochs,
        args.batch_size,
        adv_preprocess_obss,
    )

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:

        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = adv_algo.collect_experiences()
        logs2 = adv_algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [
                logs["entropy"],
                logs["value"],
                logs["policy_loss"],
                logs["value_loss"],
                logs["grad_norm"],
            ]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(
                    *data
                )
            )

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if num_frames == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {
                "num_frames": num_frames,
                "update": update,
                "model_state": policy_acmodel.state_dict(),
            }
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--model", default=None, help="name of the model")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="number of updates between two logs (default: 1)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="number of updates between two saves (default: 10, 0 means no saving)",
    )
    parser.add_argument(
        "--procs", type=int, default=1, help="number of processes (default: 16)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1000000,
        help="number of frames of training (default: 1e7)",
    )

    ## Parameters for main algorithm
    parser.add_argument(
        "--epochs", type=int, default=4, help="number of epochs for PPO (default: 4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="batch size for PPO (default: 512)"
    )
    parser.add_argument(
        "--frames-per-proc",
        type=int,
        default=256,
        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, help="discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.999,
        help="learning rate decay (default: 0.999)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss term coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="maximum norm of gradient (default: 0.5)",
    )
    parser.add_argument(
        "--optim-eps",
        type=float,
        default=1e-8,
        help="Adam and RMSprop optimizer epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--optim-alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer alpha (default: 0.99)",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="clipping epsilon for PPO (default: 0.2)",
    )
    parser.add_argument(
        "--recurrence",
        type=int,
        default=1,
        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        default=False,
        help="add a GRU to the model to handle text input",
    )
    parser.add_argument(
        "--delta", type=float, default=0.05, help="env stochasticity (default: 0.05)"
    )
    parser.add_argument(
        "--target_quantile",
        type=float,
        default=0.1,
        help="target CVaR quantile (default: 0.1)",
    )

    args = parser.parse_args()

    args.mem = args.recurrence > 1

    np.seterr(all="ignore")

    train(args)
