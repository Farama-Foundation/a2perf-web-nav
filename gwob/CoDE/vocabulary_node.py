# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A global vocabulary."""
import threading
import typing
from typing import Union

from absl import logging

Number = Union[int, float]
VocabularyElement = Union[str, Number]
import multiprocessing


class Error(Exception):
  """Base exception for vocabulary errors."""


class VocabularyOverflowError(Error):
  """Raised when the vocabulary has overflowed its max word count."""


class Vocabulary():
  """A child of a global LockedVocabulary object."""

  def __init__(self, global_vocab_node,
      max_vocabulary_size=15000):
    """Initialize the global vocabulary.

    Args:
      global_vocab_node: A reference to the global LockedVocabulary node. This
        is self-referential for the leader.
      max_vocabulary_size: The maximum size for the vocabulary.
    """
    # Local nodes will have this set. The global node will set to None.
    self._global_vocab_node = global_vocab_node

    self._max_vocabulary_size = max_vocabulary_size

    # The child's local view of the vocabulary. For the leader, this is the
    # global state.
    self._local_vocab = self._global_vocab_node.get_global_vocabulary()

  @property
  def local_vocab(self):
    return self._local_vocab

  @property
  def max_vocabulary_size(self):
    return self._max_vocabulary_size

  def __getitem__(self, key):
    """Allows dictionary access to the vocabulary."""
    return self._local_vocab[key]

  def __len__(self):
    """Gets the dictionary length of the vocabulary."""
    return len(self._local_vocab)

  def __contains__(self, item):
    """Checks if an item is in the vocabulary."""
    return item in self._local_vocab

  def add_to_vocabulary(
      self,
      words_to_add):
    """Add elements to the global vocabulary.

    Args:
      words_to_add: words to add to the vocabulary

    Returns:
      The updated vocabulary.
    """
    if not words_to_add:
      self._local_vocab = dict(self.get_global_vocabulary())
      return self._local_vocab

    with self._global_vocab_node._lock:
      self._local_vocab = dict(
          self._global_vocab_node.add_to_vocabulary(words_to_add))
    return self._local_vocab

  def get_global_vocabulary(self):
    """Return the global vocabulary."""
    return self._global_vocab_node.get_global_vocabulary()

  def save(self):
    """Overridden abstract method for saving the vocabulary object."""
    return self._global_vocab_node.save()

  def restore(self, state):
    """Overridden abstract method for restoring the vocabulary object."""
    self._global_vocab_node.restore(state)
    self._local_vocab = self._global_vocab_node.get_global_vocabulary()


class LockedVocabulary(Vocabulary):
  def __init__(self, multiprocessing_manager, max_vocabulary_size=15000,
  ):
    self._lock = multiprocessing_manager.Lock()
    self._max_vocabulary_size = max_vocabulary_size
    self._local_vocab = multiprocessing_manager.dict()
    self._next_index = 0

    super().__init__(global_vocab_node=self,
                     max_vocabulary_size=max_vocabulary_size)

  def add_to_vocabulary(
      self,
      words_to_add):
    """Add words to the global vocabulary until a specified limit.

    Args:
      words_to_add: A list of words to add to the global vocabulary.

    Returns:
      The updated global vocabulary.

    Raises:
      VocabularyOverflowError: Raised when the vocabulary has exceeded its max
        size.
    """
    for word in words_to_add:
      if word not in self._local_vocab:
        if len(self._local_vocab) >= self._max_vocabulary_size:
          raise VocabularyOverflowError(
              'Vocabulary size exceeded max size of {}.'.format(
                  self._max_vocabulary_size))
        self._local_vocab[word] = self._next_index
        self._next_index += 1
    return self._local_vocab

  def get_global_vocabulary(self):
    """Return the global vocabulary."""
    return self._local_vocab

  def save(self) -> typing.Dict[str, int]:
    """Overridden abstract method for saving the LockedVocabulary object."""
    return dict(self._local_vocab)

  def restore(self, state: multiprocessing.Manager().dict()):
    """Overridden abstract method for restoring the LockedVocabulary object."""
    self._local_vocab = state
    self._global_vocab_node = self
    max_index = max(self._local_vocab.values()) + 1 if self._local_vocab else 0
    self._next_index = max_index
