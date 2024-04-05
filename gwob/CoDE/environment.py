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
  """Web Navigation Environment."""

  def __init__(
      self,
      seed: int,
      data_dir: str,
      num_websites: int,
      difficulty: Optional[int] = None,
      designs: Optional[list[dict[str, Any]]] = None,
      global_vocabulary=None,
      use_legacy_reset: bool = False,
      use_legacy_step: bool = False,
      render_mode: str = 'image',
      **kwargs,
  ):
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

  def step(self, action, raw_state=False):
    obs, rew, terminated, truncated, info = super().step(action,
                                                         raw_state=raw_state)
    self._prev_obs = obs
    if self._use_legacy_step:
      return obs, rew, (terminated or truncated), info
    else:
      return obs, rew, terminated, truncated, info

  def reset(
      self,
      raw_state: bool = False,
      seed: int | None = None,
      options: dict[str, Any] | None = None,
  ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
    """Reset the environment."""
    design_to_use = self._sample_design()
    self._design_environment(env_design=design_to_use)
    obs, info = super().reset(raw_state=raw_state)
    if self._use_legacy_reset:
      return obs
    else:
      return obs, info

  def _design_environment(self, env_design):
    """Design the environment based` on the environment design."""
    self.current_design = env_design
    self.current_website = website_util.Website(design=env_design)
    self.design_environment(env_design=env_design, auto_num_pages=True)

  def _load_designs(self, difficulty):
    """Load the designs for the corresponding difficulty level."""

    # Load the designs file
    design_path = os.path.join(self.data_dir, 'difficulty_levels',
                               f'{difficulty:02d}.json')
    if not os.path.isfile(design_path):
      logging.info('Could not find %s', design_path)
      zip_path = os.path.join(self.data_dir, 'difficulty_levels.zip')
      if not os.path.isfile(zip_path):
        raise FileNotFoundError(f'Neither {design_path} nor {zip_path} found.'
                                f' Please make sure that the a2perf package is'
                                f' installed correctly.')

      tmp_dir = os.path.join(os.path.expanduser('~'), '.web_navigation')
      if os.path.isdir(tmp_dir):
        logging.info('Found existing tmp web_navigation directory %s', tmp_dir)
        design_path = os.path.join(tmp_dir, 'difficulty_levels',
                                   f'{difficulty:02d}.json')
        if os.path.isfile(design_path):
          logging.info('Found existing design file %s', design_path)
          with open(design_path, 'r') as f:
            return json.load(f)

      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Unzip the files here since we may not have writer permissions
        os.makedirs(tmp_dir, exist_ok=True)
        logging.info('Unzipping website design files to %s', tmp_dir)
        zip_ref.extractall(tmp_dir)

        # After unzipping, the data directory should change
        self.data_dir = os.path.join(tmp_dir, 'difficulty_levels')
        design_path = os.path.join(self.data_dir,
                                   f'{difficulty:02d}.json')

    # load the design path using JSON
    with open(design_path, 'r') as f:
      return json.load(f)

  def _sample_design(self):
    """Sample a design from the design space."""
    design = self._random.choice(self._designs)
    return design
