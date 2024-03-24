import os

import gin
import gym as legacy_gym
import gymnasium as gym
import pkg_resources

base_url_path = pkg_resources.resource_dir('a2perf.domains.web_navigation.gwob')
config_file_path = pkg_resources.resource_filename(
    'a2perf', 'domains/web_navigation/configs/web_navigation_env_config.gin'
)
base_url = f'file://{base_url_path}/'

data_directory = pkg_resources.resource_dir(
    'a2perf.domains.web_navigation.gwob.CoDE.environment')
data_dir = os.path.join(data_directory, 'data')

gin.parse_config_files_and_bindings(
    [config_file_path], None, finalize_config=False
)

gym.envs.register(
    id='WebNavigation-v0',
    entry_point=(
        'a2perf.domains.web_navigation.gwob.CoDE.environment:WebNavigationEnv'
    ),
    apply_api_compatibility=False,
    disable_env_checker=False,
    kwargs=dict(
        use_legacy_step=False,
        use_legacy_reset=False,
        data_dir=data_dir,
        base_url=base_url),
)

legacy_gym.envs.register(
    id='WebNavigation-v0',
    entry_point=(
        'a2perf.domains.web_navigation.gwob.CoDE.environment:WebNavigationEnv'
    ),
    kwargs=dict(
        use_legacy_step=True,
        use_legacy_reset=True,
        data_dir=data_dir,
        base_url=base_url),
)
