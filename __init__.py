import gin
import gym

gin.parse_config_files_and_bindings(["../domains/web_nav/web_nav/configs/envdesign.gin"], None)

gym.envs.register(
        id='WebNavigation-v0',
        entry_point='rl_perf.domains.web_nav.web_nav.environment:WebNavigationEnv',
        kwargs=
        {'base_url': 'file:///web_nav/gwob/',  # TODO: change to environment variable
         'difficulty': 'easy',
         'seed': 0,
         }
        )
