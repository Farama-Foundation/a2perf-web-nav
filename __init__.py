import os

import gin
import gymnasium

config_file_path = os.path.join(os.path.dirname(__file__), 'configs', 'web_navigation_env_config.gin')

# Add a default data directory to the gin config
data_dir = os.path.join(os.path.dirname(__file__), 'environment_generation', 'data')

gin.parse_config_files_and_bindings([config_file_path], None, finalize_config=False)
gin.parse_config(f'environment.WebNavigationEnv.data_dir = "{data_dir}"')
gym.envs.register(
    id='WebNavigation-v0',
    entry_point='rl_perf.domains.web_nav.gwob.CoDE.environment:WebNavigationEnv',
)
