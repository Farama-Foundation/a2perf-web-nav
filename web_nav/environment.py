import os

import gin
import numpy as np
import selenium
from absl import logging
from miniwob import reward

from rl_perf.domains.web_nav.CoDE import vocabulary_node
from rl_perf.domains.web_nav.CoDE import web_environment, utils


@gin.configurable('WebNavigationEnv')
class WebNavigationEnv(web_environment.GMiniWoBWebEnvironment):
    """Web Navigation Environment."""

    def __init__(
            self,
            seed,
            difficulty,
            data_dir,
            global_vocabulary=vocabulary_node.LockedVocabulary(),
            **kwargs
            ):
        super().__init__(seed=seed, global_vocabulary=global_vocabulary, **kwargs)
        self.data_dir = data_dir
        self.difficulty = difficulty
        self.browser_kwargs = kwargs['kwargs_dict']
        self._designs = self._load_designs(self.difficulty)
        self.current_design = None
        self._prev_obs = None

    def restart_browser(self):
        """Restart the browser and reset the environment."""
        # Call close on the base miniwob environment
        self.close()

        # Restart the underlying browser
        del self._wob_env.instances[0].driver
        self._wob_env.instances[0].create_driver()

        states = [None] * len(self._wob_env.instances)
        seeds = [self._random.randint(0, 1000) for _ in self._wob_env.instances]
        self._wob_env.instances[0].reset(seed=seeds, states=states)

    def step(self, action, raw_state=False):
        try:
            obs, rew, done, info = super().step(action, raw_state=raw_state)
            self._prev_obs = obs
        except selenium.common.exceptions.WebDriverException as e:
            logging.warning('Chrome crashed, restarting browser and resetting environment.')
            self.restart_browser()
            done = True
            info = {'crash': True}
            rew = 0
            return self._prev_obs, rew, done, info
        return obs, rew, done, info

    def reset(self, raw_state=False, design=None):
        """Reset the environment."""
        try:
            self._design_environment(env_design=self._sample_design())
            data = super().reset(raw_state=raw_state)
        except selenium.common.exceptions.WebDriverException as e:
            logging.info('Chrome crashed, restarting browser and resetting environment.')
            self.restart_browser()
            data = self.reset()
        return data

    def _design_environment(self, env_design):
        """Design the environment based` on the environment design."""
        self.current_design = env_design
        self.design_environment(env_design=env_design, auto_num_pages=True)

    def _load_designs(self, difficulty):
        """Load the designs for the corresponding difficulty level."""
        design_path = os.path.join(self.data_dir, f'design_{difficulty}.npy')
        if not os.path.isfile(design_path):
            raise ValueError(f'No design file found for difficulty {difficulty}')
        return np.load(design_path, allow_pickle=True)

    def _sample_design(self):
        """Sample a design from the design space."""
        return self._random.choice(self._designs)
