import gym
from CoDE import vocabulary_node

gym.envs.register(
        id='WebNavigation-v0',
        entry_point='web_nav.environment:WebNavigationEnv',
        kwargs=
        {'base_url': 'file:///web_nav/gwob/',  # TODO: change to environment varaible
         'difficulty': 'easy',
         'seed': 0,
         }
        )
