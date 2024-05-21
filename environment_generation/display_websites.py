"""After generating websites, use this script to view them in a browser."""
import functools

from absl import flags
from a2perf.domains.tfa import suite_gym

_NUM_DIFFICULTY_LEVELS = flags.DEFINE_integer(
    'num_difficulty_levels', 3,
    'Number of difficulty levels to split websites into.'
)

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 100,
    'Number of steps to run the environment for for each difficulty level.'
)


def main(_):
  for difficulty in range(_NUM_DIFFICULTY_LEVELS.value):
    default_gym_kwargs = dict(
        difficulty=difficulty,
        num_websites=1,
        seed=0,
        browser_args=dict(
            threading=False,
            chrome_options={
                '--headless=new',
                '--no-sandbox'
            }
        )
    )

    suite_load_function = functools.partial(
        suite_gym.load,
        gym_kwargs=default_gym_kwargs
    )
    env = suite_load_function('WebNavigation-v0')

    env.reset()
    env.render()
  pass
