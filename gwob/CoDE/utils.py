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

"""Utility functions for miniwob environments."""

import collections
import re

from CoDE.vocabulary_utils import tokenize
import gin
from miniwob import action
from miniwob import environment
from miniwob import reward
from miniwob import state
import numpy as np

# See https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/text
INPUT_ELEMENTS = (
    "input_button",
    "input_checkbox",
    "input_color",
    "input_date",
    "input_datetime",
    "input_datetime-local",
    "input_email",
    "input_file",
    "input_hidden",
    "input_image",
    "input_month",
    "input_number",
    "input_password",
    "input_radio",
    "input_range",
    "input_reset",
    "input_search",
    "input_submit",
    "input_tel",
    "input_text",
    "input_time",
    "input_url",
    "input_week",
    "input_password",
)

COMMON_ELEMENTS = ("button", "label", "select")
ADDITIONAL_ELEMENTS = ("a", "img", "div", "span")


def create_environment(subdomain, base_url, kwargs_dict):
  """Create a miniwob environment with subdomain.

  Args:
    subdomain: Name of the environment.
    base_url: Base url that shows the location of the envionment resources.
    kwargs_dict: Additional arguments that are passed to environment
      configuration.

  Returns:
    A MiniWoBEnvironment instance for interacting with websites.
  """

  env = environment.MiniWoBEnvironment(subdomain)
  env.configure(
      num_instances=1,
      seeds=[np.random.randint(0, 1000)],
      base_url=base_url,
      reward_processor=reward.get_raw_reward,
      **kwargs_dict)
  return env


def create_element_click(element, fail_hard=True):
  """Create dom element click action.

    This allows gin configuration for failure case.

  Args:
    element: A web page element.
    fail_hard: If true, it throw an exception if the click fails.

  Returns:
    A MiniWoBElementClick action.
  """
  return action.MiniWoBElementClick(
      element, fail_hard=fail_hard, handle_select_in_selenium=False)


def create_dom_graph(observation, prune_refs=None):
  """Return dom nodes and graph from a given observation.

    Create a list of (senders, receivers) as the DOM structure representation.

  Args:
    observation: Observation from web environment.
    prune_refs: If given, these will be used to prune elements based on their
      ref attribute. This attribute is miniwob specific.

  Returns:
    A tuple of (dom elements, sender elements, receiver elements) that
    corresponds to nodes and edges of DOM graph.
  """
  dom_elems = observation.dom_elements
  if prune_refs:
    dom_elems = [elem for elem in dom_elems if elem.ref in prune_refs]
  senders, receivers = [], []
  for element in dom_elems:
    senders.append(element.ref)
    for child in element.children:
      receivers.append(child.ref)
  return dom_elems, senders, receivers


@gin.configurable
def order_dom_elements(dom_elements, only_leaf=False, html_id_prefix=None):
  """Order dom elements based on their ref.

    This is used to have a consistent order on DOM elements when selecting an
    element using an index.
  Args:
    dom_elements: A sequence of elements from DOM tree.
    only_leaf: If true, return only leaf nodes.
    html_id_prefix: If given, this will be used to select DOM elements based on
      their html ids. If html id matches the prefix, that DOM element will be
      included.

  Returns:
    A sequence of DOM elements that are sorted based on their ref.
  """
  dom_elems = []
  if html_id_prefix:
    visited = set()
    matching_elements = set()

    def traverse_and_label(element):
      # 'ref' is an attribute generated by MiniWoB. It allows us to point to
      # any element uniquely within an episode. This is not DOM or HTML related
      # feature and only specific to MiniWoB.
      if element.ref in visited:
        return element.ref in matching_elements
      matches = False
      for child in element.children:
        matches = traverse_and_label(child) | matches
      visited.add(element.ref)
      if not matches and element.id and element.id.startswith(html_id_prefix):
        matching_elements.add(element.ref)
        return True
      return False

    for elem in dom_elements:
      matches = traverse_and_label(elem)
      if matches and elem.ref and (elem.is_leaf or not only_leaf):
        dom_elems.append(elem)
  else:
    for elem in dom_elements:
      if elem.ref and (elem.is_leaf or not only_leaf):
        dom_elems.append(elem)
  return sorted(dom_elems, key=lambda x: x.ref)


@gin.configurable
def get_dom_elements(observation, prune_refs=None, prune_form_input=False):
  """Return a list of dom elements from a given observation.

  Args:
    observation: Observation from web environment.
    prune_refs: If given, these will be used to prune elements based on their
      refs. Refs are miniwob specific and might not work outside of miniwob.
    prune_form_input: If true, take only input elements such as input_text,
      input_radio or if they have text values that could be used as label of an
      input.

  Returns:
    A list of DOM elements.
  """

  def check_prune_element(elem):
    return elem.tag in INPUT_ELEMENTS + COMMON_ELEMENTS or (
        elem.tag in ADDITIONAL_ELEMENTS and (elem.text or elem.value))

  dom_elements = []

  def _get_elements(element, dom_elements):
    for child in element.children:
      if child.tag == "option":
        continue
      _get_elements(child, dom_elements)
    dom_elements.append(element)

  if not prune_form_input:
    dom_elements = observation.dom_elements
  else:
    _get_elements(observation.dom, dom_elements)
  dom_elements = order_dom_elements(dom_elements)
  if prune_form_input:
    dom_elements = [elem for elem in dom_elements if check_prune_element(elem)]
  if prune_refs:
    return [
        elem for elem in dom_elements if elem.ref and elem.ref in prune_refs
    ]
  return dom_elements


def select_dom_element(dom_elements, index):
  """Select a dom element for a given index.

  Args:
    dom_elements: A list of dom elements.
    index: An integer index pointing to a dom element.

  Returns:
    A dom element that corresponds to the given index.
  """
  # An ordered dictionary to map from element refs to dom elements.
  # This is used to ensure refs are unique and sorted.
  # Sorting aligns with order_dom_elements function.
  dom_map = collections.OrderedDict()
  for elem in order_dom_elements(dom_elements):
    dom_map[elem.ref] = elem
  if index >= len(dom_map):  # index is invalid
    return None
  dom_elem = list(dom_map.items())[index][1]
  return dom_elem


def generate_click_action(dom_elements, index_or_element):
  """Generate a click action using dom element or its index.

  Args:
    dom_elements: A list of dom elements.
    index_or_element: An integer index pointing to a dom element or a DOMElement
      instance.

  Returns:
    A miniwob click action.
  """
  if isinstance(index_or_element, state.DOMElement):
    dom_elem = index_or_element
  else:
    dom_elem = select_dom_element(dom_elements, index_or_element)
  if not dom_elem:
    return None
  click = create_element_click(dom_elem)
  return click


def generate_focus_and_type_action(dom_elements, index_or_element,
                                   type_seq, typed_refs):
  """Generate focus and type action using dom element or its index.

  Args:
    dom_elements: A list of dom elements.
    index_or_element: An integer index pointing to a dom element or a DOMElement
      instance.
    type_seq: A sequence of tokens to use with keyboard action.
    typed_refs: A set of refs that correspond to elements that are already typed
      in the current episode. This is used to check whether a space should be
      inserted before typing or not.

  Returns:
    A MiniWoBFocusAndType action that simulates typing a sequence into an
    element.
  """
  if isinstance(index_or_element, state.DOMElement):
    dom_elem = index_or_element
  else:
    dom_elem = select_dom_element(dom_elements, index_or_element)

  if not dom_elem:
    return None
  if type_seq and dom_elem.ref:
    if dom_elem.ref in typed_refs:
      type_seq = " " + type_seq
    else:
      typed_refs.add(dom_elem.ref)
  return action.MiniWoBFocusAndType(
      dom_elem, type_seq, handle_select_in_selenium=False)


def generate_web_action(dom_elements,
                        action_type,
                        index_or_dom_element,
                        type_seq="",
                        typed_refs=None):
  """Generate a click or keyboard action.

  Args:
    dom_elements: A list of dom elements.
    action_type: Type of the action. Either click (1) or keyboard (0).
    index_or_dom_element: An integer index pointing to a dom element or a
      DOMElement instance.
    type_seq: A sequence of tokens to use with keyboard action.
    typed_refs: A set of refs that correspond to elements that are already typed
      in the current episode. This is used to check whether a space should be
      inserted before typing or not.

  Returns:
    A MiniWoBClick or MiniWoBFocusAndType action that
    simulates clicking on an element or typing a sequence into an element.

  Raises:
    A ValueError if the action type is incorrect.
  """
  if action_type == 0:  # focus and keyboard
    return generate_focus_and_type_action(dom_elements, index_or_dom_element,
                                          type_seq, typed_refs)
  elif action_type == 1:  # click
    return generate_click_action(dom_elements, index_or_dom_element)
  raise ValueError(
      "Web action should be 0 (focus-and-keyboard) or 1 (click) but found {}"
      .format(action_type))


@gin.configurable
def dom_attributes(dom_element, num_attributes=-1):
  """Return a list of dom attributes.


  Args:
    dom_element: A dom element.
    num_attributes: If given, use this many attributes that are sorted according
      to their human readability and importance.

  Returns:
    A list of attributes that represents the dom element.
  """
  result = [
      dom_element.tag, dom_element.value, dom_element.text,
      dom_element.placeholder, dom_element.classes
  ]
  if num_attributes <= 0 or num_attributes > len(result):
    return result
  return result[0:num_attributes]


def dom_element_representation(dom_element,
                               local_vocabulary,
                               dom_attribute_sequence_length,
                               num_attributes=-1):
  """Tokenize and indexify a dom element.

  Args:
    dom_element: A dom element instance.
    local_vocabulary: A dictionary to map tokens to ids.
    dom_attribute_sequence_length: Maximum allowed length of an attribute
      sequence.
    num_attributes: If given, use this many attributes that are sorted according
      to their human readability and importance.

  Returns:
    A tuple of (dom, mask) where dom is the ids of attributes for each dom
    element and mask is the masking for attribute sequences.
  """
  dom_element_attributes = dom_attributes(
      dom_element, num_attributes=num_attributes)
  tokenized_attributes = [
      tokenize(dom_element_attr) for dom_element_attr in dom_element_attributes
  ]
  indexified_attributes = [
      indexify(attr, local_vocabulary, dom_attribute_sequence_length)
      for attr in tokenized_attributes
  ]
  mask = np.asarray([[0] * dom_attribute_sequence_length
                     for _ in range(len(tokenized_attributes))],
                    dtype="float")
  for i, dom_element_attr in enumerate(tokenized_attributes):
    mask[i, 0:len(dom_element_attr)] = 1.0
  return indexified_attributes, mask


@gin.configurable
def get_word_to_id(word,
                   local_vocabulary,
                   check_for_numbers=False,
                   token_length=-1):
  """Get id of a word.

    This function explicitly extracts numbers from the word. If there are
    multiple words, it returns the first occurrence.
    If a token is not in the vocabulary, it is added to it as long as the size
    of the vocabulary is less than the allowed amount.

  Args:
    word: A single token.
    local_vocabulary: Mapping from tokens to ids.
    check_for_numbers: If true, check the availability of a number in the token
      and use that.
    token_length: If given, tokens longer than this threshold will not be added
      to the vocabulary.

  Returns:
    A single integer id that corresponds to the word.
  """
  max_vocab_size = local_vocabulary.max_vocabulary_size

  if word in local_vocabulary:
    return local_vocabulary[word]
  elif str(word) in local_vocabulary:
    return local_vocabulary[str(word)]
  elif token_length > 0 and len(word) > token_length:
    return local_vocabulary["OOV"]
  elif check_for_numbers:
    try:
      raw_numbers = re.findall(r"\d+", word)
      if raw_numbers:
        if raw_numbers[0] not in local_vocabulary:
          if max_vocab_size > 0 and len(local_vocabulary) >= max_vocab_size:
            return local_vocabulary["OOV"]
          local_vocabulary.add_to_vocabulary([raw_numbers[0]])
        return local_vocabulary[raw_numbers[0]]
    except TypeError as _:
      pass
    except ValueError as _:
      pass
    except AttributeError as _:
      pass
  # some numbers (ui-id-NUMBER) from autocomplete will be OOV .
  if max_vocab_size != -1 and len(local_vocabulary) >= max_vocab_size:
    return local_vocabulary["OOV"]
  local_vocabulary.add_to_vocabulary([str(word)])
  return get_word_to_id(
      word, local_vocabulary, check_for_numbers=check_for_numbers)


@gin.configurable
def indexify(tokens, local_vocabulary, max_sequence_length):
  """Map tokens to ids and pad with id of NULL if necessary.

  Args:
    tokens: A sequence of tokens. If not, assume a single token.
    local_vocabulary: A dictionary of word to id mapping.
    max_sequence_length: Maximum allowed sequence length.

  Returns:
    A sequence of ids that corresponds to the input sequence of tokens.
  """
  if not isinstance(tokens, list):
    tokens = [tokens]
  indices = [get_word_to_id(s, local_vocabulary) for s in tokens]
  if len(indices) > max_sequence_length:
    return indices[:max_sequence_length]
  pad_length = max_sequence_length - len(indices)
  return indices + [local_vocabulary["NULL"]] * pad_length
