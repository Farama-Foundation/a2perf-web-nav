from __future__ import annotations

import json
import os
import zipfile
from typing import Any
from typing import Optional

import gin
from absl import logging

from a2perf.domains.web_navigation.environment_generation import website_util
from a2perf.domains.web_navigation.gwob.CoDE import web_environment


@gin.configurable('WebNavigationEnv')
class WebNavigationEnv(web_environment.GMiniWoBWebEnvironment):
  """
  The gym environment for web navigation tasks.

  Attributes:
      data_dir (str): Directory path where the data is stored.
      num_websites (int): Number of websites.
      difficulty (Optional[int]): Difficulty level of the environment.
      designs (Optional[list[dict[str, Any]]]): List of website designs.
      global_vocabulary: Global vocabulary for the environment.
      use_legacy_reset (bool): Whether to use legacy reset method.
      use_legacy_step (bool): Whether to use legacy step method.
      render_mode (str): Rendering mode.
      raw_state (bool): Whether to return raw state.
      browser_kwargs: Browser arguments.

  Methods:
      step(action): Take a step in the environment.
      reset(seed, options): Reset the environment.
      _design_environment(env_design): Design the environment based on the environment design.
      _load_designs(difficulty): Load the designs for the corresponding difficulty level.
      _sample_design(): Sample a design from the design space.
  """

  def __init__(
      self,
      seed: int,
      data_dir: str,
      num_websites: int | None = None,
      difficulty: Optional[int] = None,
      designs: Optional[list[dict[str, Any]]] = None,
      global_vocabulary=None,
      use_legacy_reset: bool = False,
      use_legacy_step: bool = False,
      render_mode: str = 'image',
      raw_state: bool = False,
      **kwargs,
  ):
    """
    Initializes a WebNavigationEnv object.

    Args:
        seed (int): The seed for random number generation.
        data_dir (str): Directory path where the data is stored.
        num_websites (int, optional): Number of websites. Defaults to None.
        difficulty (Optional[int], optional): Difficulty level of the environment. Defaults to None.
        designs (Optional[list[dict[str, Any]]], optional): List of website designs. Defaults to None.
        global_vocabulary: Global vocabulary for the environment. Defaults to None.
        use_legacy_reset (bool, optional): Whether to use legacy reset method. Defaults to False.
        use_legacy_step (bool, optional): Whether to use legacy step method. Defaults to False.
        render_mode (str, optional): Rendering mode. Defaults to 'image'.
        raw_state (bool, optional): Whether to return raw state. Defaults to False.

    Raises:
        ValueError: If num_websites is not specified or if both designs and difficulty are specified.
    """
    super().__init__(seed=seed, global_vocabulary=global_vocabulary,
                     render_mode=render_mode, **kwargs)
    self.data_dir = data_dir
    self.difficulty = difficulty
    self.num_websites = num_websites
    self.current_website = None
    self._use_legacy_reset = use_legacy_reset
    self._use_legacy_step = use_legacy_step
    if kwargs is not None:
      self.browser_kwargs = kwargs.get('browser_args', None)
    else:
      self.browser_kwargs = None

    if num_websites is None :
      raise ValueError('Number of websites (num_websites) must be specified.')

    if (designs is not None and difficulty is not None):
      raise ValueError('Either designs or difficulty must be specified, but '
                       'not both.')
    if not (designs or difficulty):
      raise ValueError('Either designs or difficulty must be specified.')

    if designs is None:
      if not (1 <= difficulty <= 3):
        raise ValueError(f'Difficulty must be between 1 and 3, but got '
                         f'{difficulty}.')
      designs = self._load_designs(self.difficulty)

    self._designs = designs

    # Make sure that num_websites is not greater than the number of designs
    if self.num_websites > len(self._designs):
      raise ValueError(f'Number of websites to sample ({self.num_websites}) '
                       f'cannot be greater than the number of designs '
                       f'({len(self._designs)}).')

    # Randomly sample num_websites websites from the designs
    self._designs = self._random.choice(a=self._designs, size=self.num_websites,
                                        replace=False)

    self._current_design = None
    self._prev_obs = None

    self._raw_state = raw_state

  def step(self, action):
    """
    Take a step in the environment.

    Args:
        action: The action to take. The action should be passed as a scalar. Two types of actions are possible. Firstly, abstract navigation allows to directly refer to an element, and the profile is irrelevant. In this case, the action is converted to a tuple. If abstract navigation is desired, we have to pass `use_conceptual=True` when initializing the environment. Secondly, the action can refer to a pair of elements and profile fields. The agent will then enter the value of the profile key corresponding to the selected DOM element.

    Returns:
        tuple: Observation, reward, termination status, truncation status, info.
    """
    obs, rew, terminated, truncated, info = super().step(action,
                                                         raw_state=self._raw_state)
    self._prev_obs = obs
    if self._use_legacy_step:
      return obs, rew, (terminated or truncated), info
    else:
      return obs, rew, terminated, truncated, info

  def reset(
      self,
      seed: int | None = None,
      options: dict[str, Any] | None = None,
  ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
    """
    Reset the environment.

    Args:
        seed (int, optional): The seed for random number generation. Defaults to None.
        options (dict[str, Any], optional): Options for resetting the environment. Defaults to None.

    Returns:
        tuple: Observation and info.
    """
    design_to_use = self._sample_design()
    self._design_environment(env_design=design_to_use)
    obs, info = super().reset(raw_state=self._raw_state)
    if self._use_legacy_reset:
      return obs
    else:
      return obs, info

  def _design_environment(self, env_design):
    """
    Design the environment based on the environment design.

    Args:
        env_design: The environment design.
    """
    self.current_design = env_design
    self.current_website = website_util.Website(design=env_design)
    self.design_environment(env_design=env_design, auto_num_pages=True)

  def _load_designs(self, difficulty):
    """
    Load the designs for the corresponding difficulty level.

    Args:
        difficulty: The difficulty level.

    Returns:
        list: List of website designs.
    """
    design_path = os.path.join(self.data_dir, f'{difficulty:02d}.json')
    with open(design_path, 'r') as f:
      return json.load(f)

  def _sample_design(self):
    """Sample a design from the design space."""
    design = self._random.choice(self._designs)
    return design
