import os

import gin
import gym

config_file_path = os.path.join(os.path.dirname(__file__), 'web_nav', 'configs', 'envdesign.gin')
gin.parse_config_files_and_bindings([config_file_path], None, finalize_config=False)

gym.envs.register(
        id='WebNavigation-v0',
        entry_point='rl_perf.domains.web_nav.web_nav.environment:WebNavigationEnv',
        kwargs=
        {'base_url': 'file:///web_nav/gwob/',  # TODO: change to environment variable
         'difficulty': 'easy',
         'seed': 0,
         }
        )
