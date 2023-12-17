import json
import os
import zipfile

import gin
from absl import logging

from a2perf.domains.web_navigation.environment_generation import website_util
from a2perf.domains.web_navigation.gwob.CoDE import web_environment


@gin.configurable('WebNavigationEnv')
class WebNavigationEnv(web_environment.GMiniWoBWebEnvironment):
  """Web Navigation Environment."""

  def __init__(
      self,
      seed,
      data_dir,
      difficulty=None,
      designs=None,
      global_vocabulary=None,
      **kwargs,
  ):
    super().__init__(seed=seed, global_vocabulary=global_vocabulary, **kwargs)
    self.data_dir = data_dir
    self.difficulty = difficulty
    self.browser_kwargs = kwargs['browser_args']
    assert (designs is None) != (difficulty is None), (
        'Either designs or difficulty must be specified, but not both.')

    self._designs = []
    if designs is None:
      designs = self._load_designs(self.difficulty)

    for design in designs:
      self._designs.append(website_util.Website(design=design))

    self._current_design = None
    self._prev_obs = None

  def step(self, action, raw_state=False):
    obs, rew, done, info = super().step(action, raw_state=raw_state)
    self._prev_obs = obs
    return obs, rew, done, info

  def reset(self, raw_state=False):
    """Reset the environment."""
    self._design_environment(env_design=self._sample_design())
    data = super().reset(raw_state=raw_state)
    return data

  def _design_environment(self, env_design):
    """Design the environment based` on the environment design."""
    self.current_design = env_design
    self.design_environment(env_design=env_design, auto_num_pages=True)

  def _load_designs(self, difficulty):
    """Load the designs for the corresponding difficulty level."""
    design_path = os.path.join(self.data_dir,
                               f'{difficulty:02d}.json')
    if not os.path.isfile(design_path):
      logging.info('Could not find %s', design_path)
      zip_path = os.path.join(self.data_dir, 'difficulty_levels.zip')
      if not os.path.isfile(zip_path):
        raise FileNotFoundError(f'Could not find {zip_path}')

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
    website = self._random.choice(self._designs)
    return website.to_design()
