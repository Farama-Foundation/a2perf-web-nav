import random

import gin
from CoDE import web_environment
from CoDE import vocabulary_node


@gin.configurable('WebNavigationEnv')
class WebNavigationEnv(web_environment.GMiniWoBWebEnvironment):
    """Web Navigation Environment."""

    def __init__(
            self,
            seed,
            difficulty,
            global_vocabulary=vocabulary_node.LockedVocabulary(),
            **kwargs
            ):
        kwargs['global_vocabulary'] = global_vocabulary
        super().__init__(**kwargs)

        self.seed = seed
        if seed is None:
            self.seed = 0
        self.rng = random.Random(self.seed)

        self.difficulty = difficulty
        if self.difficulty not in ['easy', 'medium', 'hard']:
            raise ValueError('difficulty must be easy, medium, or hard')

        self._designs = self._load_designs(self.difficulty)
        self.design_environment(env_design=self._sample_design())

    def reset(self, raw_state=False):
        """Reset the environment."""
        self._design_environment(env_design=self._sample_design())
        return super().reset(raw_state=raw_state)

    def _design_environment(self, env_design):
        """Design the environment based on the environment design."""
        self.design_environment(env_design=env_design, auto_num_pages=True)

    def _load_designs(self, difficulty):
        """Load the designs for the corresponding difficulty level."""
        return [{'number_of_pages': 5,
                 'action': [1, 2, 3, 4, 5],
                 'action_page': [0, 1, 2, 3, 4]}
                ]

    def _sample_design(self):
        """Sample a design from the design space."""
        return self.rng.choice(self._designs)
