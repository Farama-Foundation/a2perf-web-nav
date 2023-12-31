import os

import gin
import gym as legacy_gym
import gymnasium as gym

config_file_path = os.path.join(
    os.path.dirname(__file__), 'configs', 'web_navigation_env_config.gin'
)
base_url_path = os.path.join(os.path.dirname(__file__), 'gwob')
base_url = f'file://{base_url_path}/'

# Add a default data directory to the gin config
data_dir = os.path.join(
    os.path.dirname(__file__),
    'environment_generation',
    'data',
)

gin.parse_config_files_and_bindings(
    [config_file_path], None, finalize_config=False
)
gym.envs.register(
    id='WebNavigation-v0',
    entry_point=(
        'a2perf.domains.web_navigation.gwob.CoDE.environment:WebNavigationEnv'
    ),
    apply_api_compatibility=False,
    disable_env_checker=True,
    kwargs=dict(data_dir=data_dir, base_url=base_url),
)

# Some RL frameworks still use the legacy gym interface, so we register the environment there as well.
legacy_gym.envs.register(
    id='WebNavigation-v0',
    entry_point=(
        'a2perf.domains.web_navigation.gwob.CoDE.environment:WebNavigationEnv'
    ),
    kwargs=dict(data_dir=data_dir, base_url=base_url),
)
