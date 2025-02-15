import torch

import utils
from model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(
        self,
        acmodel,
        obs_space,
        model_dir,
        device=None,
        argmax=False,
        num_envs=1,
        pretrained=False,
    ):
        obs_space, self.preprocess_obss = utils.get_policy_obss_prepocessor(obs_space)
        self.acmodel = acmodel
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(
                self.num_envs, self.acmodel.memory_size, device=self.device
            )

        if pretrained:
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(
                dones, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])


class AdversaryAgent:
    """An adversary agent.

    It is able:
    - to perturbate probability transitions,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(
        self,
        acmodel,
        env,
        model_dir,
        device=None,
        num_envs=1,
        pretrained=False,
    ):
        _, self.preprocess_obss = utils.get_adversary_obss_preprocessor(
            env.observation_space
        )
        self.acmodel = acmodel
        self.device = device
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(
                self.num_envs, self.acmodel.memory_size, device=self.device
            )

        if pretrained:
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)

    def get_perturbations(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                _, dist, _, self.memories = self.acmodel(
                    preprocessed_obss, self.memories
                )
            else:
                _, dist, _ = self.acmodel(preprocessed_obss)

        perturbations = dist.mean

        return perturbations.cpu().numpy()

    def get_perturbation(self, obs):
        return self.get_perturbations([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(
                dones, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
