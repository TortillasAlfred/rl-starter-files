import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList(
                {
                    "image": preprocess_images(obss, device=device),
                    "transition_probas": preprocess_transitions(obss, device=device),
                }
            )

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == [
        "image"
    ]:
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList(
                {
                    "image": preprocess_images(
                        [obs["image"] for obs in obss], device=device
                    ),
                    "transition_probas": preprocess_transitions(
                        [obs["transition_probas"] for obs in obss], device=device
                    ),
                }
            )

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def get_policy_obss_prepocessor(obs_space):
    obs_space = {"state": (4,)}

    def preprocess_obss(obss, device=None):
        return torch_ac.DictList(
            {
                "state": preprocess_states(
                    [obs["position"] for obs in obss],
                    [obs["direction"] for obs in obss],
                    device=device,
                )
            }
        )

    return obs_space, preprocess_obss


def get_adversary_obss_preprocessor(obs_space):
    obs_space = {"state": (80,)}

    def preprocess_obss(obss, device=None):
        return torch_ac.DictList(
            {
                "transition_probas": preprocess_transitions(
                    [obs["transition_probas"] for obs in obss], device=device
                ),
                "remaining_budget": torch.tensor(
                    [obs["remaining_budget"] for obs in obss],
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(-1),
            }
        )

    return obs_space, preprocess_obss


def preprocess_states(positions, directions, device=None):
    positions = torch.tensor(positions, device=device, dtype=torch.float32)
    directions = torch.tensor(directions, device=device, dtype=torch.float32)

    states = torch.hstack((positions, directions))

    return states


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_transitions(transitions, device=None):
    transitions = torch.stack(transitions)
    transitions = transitions.to(device).type(dtype=torch.float)

    return transitions


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, : len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
