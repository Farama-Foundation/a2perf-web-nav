import gin
import gym
from absl import app
import web_nav

from rl_perf.domains.web_nav.CoDE import utils
from rl_perf.domains.web_nav.CoDE import vocabulary_node
from rl_perf.domains.web_nav.CoDE import web_environment
from rl_perf.domains.web_nav.CoDE import web_primitives


def main(_):
    # Create the environment.
    gin.parse_config_files_and_bindings(["./web_navigation/configs/envdesign.gin"], None)
    env = gym.make('WebNavigation-v0')
    print(env)


if __name__ == '__main__':
    app.run(main)
