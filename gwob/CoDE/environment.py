import os
import pickle

import gin

from rl_perf.domains.web_nav.environment_generation import website_util
from rl_perf.domains.web_nav.gwob.CoDE import vocabulary_node
from rl_perf.domains.web_nav.gwob.CoDE import web_environment


@gin.configurable('WebNavigationEnv')
class WebNavigationEnv(web_environment.GMiniWoBWebEnvironment):
    """Web Navigation Environment."""

    def __init__(
            self,
            seed,
            difficulty,
            data_dir,
            designs=None,
            global_vocabulary=vocabulary_node.LockedVocabulary(),
            **kwargs,
    ):
        super().__init__(seed=seed, global_vocabulary=global_vocabulary, **kwargs)
        self.data_dir = data_dir
        self.difficulty = difficulty
        self.browser_kwargs = kwargs['kwargs_dict']

        if designs is None:
            self._designs = self._load_designs(self.difficulty)
        else:
            design_objs = []
            for design in designs:
                design_objs.append(website_util.Website(design=design))
            self._designs = design_objs

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
        design_path = os.path.join(self.data_dir, f'design_{difficulty}.pkl')
        if not os.path.isfile(design_path):
            raise ValueError(f'No design file found for difficulty {difficulty}')
        # return np.load(design_path, allow_pickle=True)
        # load the design path using pickle
        with open(design_path, 'rb') as f:
            return pickle.load(f)

    def _sample_design(self):
        """Sample a design from the design space."""
        website = self._random.choice(self._designs)
        return website.to_design()
