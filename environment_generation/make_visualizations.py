from absl import app

# noinspection PyUnresolvedReferences
from a2perf.domains import web_navigation

import gymnasium as gym
def main(_):
  env = gym.make('WebNavigation-v0',
                 difficulty=1,
                 browser_args=dict(
                     threading=False,
                     chrome_options={
                         '--headless',
                         '--disable-gpu',
                         '--disable-dev-shm-usage',
                         '--no-sandbox',
                         '--remote-debugging-port=9222',
                     }
                 )
  pass


if __name__ == '__main__':
  app.run(main)
