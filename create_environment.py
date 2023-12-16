import gin
import gym
from absl import app
import web_nav

from a2perf.domains.web_navigation.CoDE import utils
from a2perf.domains.web_navigation.CoDE import vocabulary_node
from a2perf.domains.web_navigation.CoDE import web_environment
from a2perf.domains.web_navigation.CoDE import web_primitives


def main(_):
    # Create the environment.
    gin.parse_config_files_and_bindings(["./web_navigation/configs/envdesign.gin"], None)
    env = gym.make('WebNavigation-v0')
    print(env)


if __name__ == '__main__':
    app.run(main)
