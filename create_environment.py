import gin
import gym
from absl import app
import web_nav

from CoDE import utils
from CoDE import vocabulary_node
from CoDE import web_environment
from CoDE import web_primitives


def main(_):
    # Create the environment.
    gin.parse_config_files_and_bindings(["./web_nav/configs/envdesign.gin"], None)
    env = gym.make('WebNavigation-v0')
    print(env)


if __name__ == '__main__':
    app.run(main)
