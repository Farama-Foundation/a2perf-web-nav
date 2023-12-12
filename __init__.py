import gin
import gym as legacy_gym
import gymnasium as gym
import os


target_path = os.path.join('file:///',os.path.dirname(__file__).strip('/'),'gwob/')

config_file_path = os.path.join(os.path.dirname(__file__), 'configs', 'web_navigation_env_config.gin')

# Add a default data directory to the gin config
data_dir = os.path.join(os.path.dirname(__file__), 'environment_generation', 'data')


gin.parse_config_files_and_bindings([config_file_path], None,
                                    finalize_config=False)
gin.parse_config(f'environment.WebNavigationEnv.data_dir="{data_dir}"')

gym.envs.register(
    id='WebNavigation-v0',
    apply_api_compatibility=True,
    entry_point='rl_perf.domains.web_nav.gwob.CoDE.environment:WebNavigationEnv',
    kwargs={
        'base_url': 
            target_path,
}
)

legacy_gym.envs.register(
    id='WebNavigation-v0',
    apply_api_compatibility=True,
    entry_point='rl_perf.domains.web_nav.gwob.CoDE.environment:WebNavigationEnv',
    kwargs={
        'base_url': 
            target_path,
}
)

