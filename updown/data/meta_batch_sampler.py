# coding=utf-8
import numpy as np
import torch

from torch.utils.data import Dataset
from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.readers import CocoCaptionsReader

from updown.utils.constraints import ConstraintFilter, FiniteStateMachineBuilder

from collections import Counter, defaultdict

import json
from typing import Any, Dict, List, Tuple, Union

import h5py
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from torch.utils.data import Sampler


class MetaBatchSampler(Sampler):
    r'''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, vocabulary,
                 captions_jsonpath,
                 classes_per_it,
                 num_samples,
                 iterations,
                 max_caption_length=20,
                 in_memory=False):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        self.vocabulary = vocabulary
        self.captions_jsonpath = captions_jsonpath
        self.max_caption_length = max_caption_length
        self.classes_per_it = classes_per_it
        self.samples_per_class = num_samples
        self.iterations = iterations

        with open(self.captions_jsonpath) as cap:
            captions_json: Dict[str, Any] = json.load(cap)

        PUNCTUATIONS: List[str] = [
            "''", "'", "``", "`", "(", ")", "{", "}",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"
        ]

        self.n_data = len(captions_json["annotations"])
        self.words2idxs = defaultdict(list)
        print(f"Setting up sampler...")
        i = 0
        for caption_item in tqdm(captions_json["annotations"]):
            caption: str = caption_item["caption"].lower().strip()
            caption_tokens: List[str] = word_tokenize(caption)
            caption_tokens = list(set([ct for ct in caption_tokens if ct not in PUNCTUATIONS]))
            for ct in caption_tokens:
                self.words2idxs[ct] += [i]
            i+=1

        self.classes = list(self.words2idxs.keys())

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config
        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            captions_jsonpath=_C.DATA.TRAIN_CAPTIONS,
            classes_per_it=_C.DATA.CLASSES_PER_IT,
            num_samples=_C.DATA.NUM_SAMPLES,
            iterations=_C.DATA.ITERATIONS,
            max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
            in_memory=kwargs.pop("in_memory"),
        )

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))
            i=0
            for idx in c_idxs:
                c = self.classes[idx]
                possible = self.words2idxs[c]
                if len(possible) >= spc:
                    s = slice(i * spc, (i + 1) * spc)

                    sample_idxs = torch.randperm(possible)[:spc]
                    batch[s] = sample_idxs
                    i+=1
                if i > cpi:
                    break

            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        # Number of training examples are number of captions, not number of images.
        return len(self.n_data)





