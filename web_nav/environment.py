import random
import os
import gin
import numpy as np
from rl_perf.domains.web_nav.CoDE import web_environment
from rl_perf.domains.web_nav.CoDE import vocabulary_node


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
        kwargs['global_vocabulary'] = global_vocabulary
        # kwargs['kwargs_dict'] = {'seed': seed}
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.difficulty = difficulty
        self._designs = self._load_designs(self.difficulty)

    def reset(self, raw_state=False):
        """Reset the environment."""
        self._design_environment(env_design=self._sample_design())
        return super().reset(raw_state=raw_state)

    def _design_environment(self, env_design):
        """Design the environment based on the environment design."""
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
