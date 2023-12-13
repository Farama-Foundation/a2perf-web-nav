import collections
import copy
import multiprocessing
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
from rl_perf.domains.web_nav.environment_generation import website_util
from rl_perf.domains.web_nav.gwob.CoDE import web_primitives

# Define flags
_SEED = flags.DEFINE_integer('seed', 1, 'Random seed for web page generation')
_NUM_PAGES = flags.DEFINE_integer(
    'num_pages', 10000, 'Initial number of pages to generate.'
)
_MAX_NUM_PRIMITIVES = flags.DEFINE_integer(
    'max_num_primitives', 25, 'Maximum number of primitives per page.'
)
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', 0, 'Number of processes to use.'
)
_MAX_HORIZON = flags.DEFINE_integer(
    'max_horizon', 25, 'Maximum number of pages in a sequence.'
)
_WEBSITES_TO_GENERATE = flags.DEFINE_integer(
    'websites_to_generate', 40000, 'Number of websites to generate.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', './data', 'Directory to save generated websites to.'
)


def generate_website(current_website, next_website):
  # We create a new website object, then copy over the pages from the two
  new_website = website_util.Website(first_page=current_website._pages[0])
  new_website._pages = copy.copy(current_website._pages)
  new_website._pages.extend(next_website._pages)
  new_website.update()

  assert new_website._num_possible_correct_steps == (
      current_website._num_possible_correct_steps
      + next_website._num_possible_correct_steps
  ), 'Number of possible correct steps should be the sum of the two websites.'

  if new_website._num_possible_correct_steps <= _MAX_HORIZON.value:
    logging.info('Successfully generated website.')
    return [new_website]
  return []


def generate_page(page_id):
  """Generates a single page."""
  page = website_util.Page(page_id)
  num_primitives = np.random.randint(1, _MAX_NUM_PRIMITIVES.value + 1)
  for _ in range(num_primitives):
    primitive_name = np.random.choice(web_primitives.CONCEPTS[1:])
    primitive_id = web_primitives.CONCEPTS.index(primitive_name)
    primitive = website_util.generate_primitive(primitive_name, primitive_id)
    page.add_primitive(primitive)
    if page.num_possible_correct_interactions >= _MAX_HORIZON.value:
      break
  return page


def main(_):
  assert (
      _WEBSITES_TO_GENERATE.value >= _NUM_PAGES.value
  ), 'Number of pages must be less than number of websites to generate.'
  np.random.seed(_SEED.value)
  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  num_processes = _NUM_PROCESSES.value or multiprocessing.cpu_count()

  # Generate pages by placing random primitives on them.
  with multiprocessing.Pool(num_processes) as pool:
    pages = pool.map(generate_page, range(_NUM_PAGES.value))
  logging.info('Generated %d pages.', len(pages))

  with open(os.path.join(_OUTPUT_DIR.value, 'pages.pkl'), 'wb') as f:
    pickle.dump(pages, f)

  # Using pages to generate initial websites
  websites = [website_util.Website(p) for p in pages]

  # Generate new websites
  combined_websites = []
  while len(combined_websites) < _WEBSITES_TO_GENERATE.value:
    # Generate pairs of websites
    pairs = [
        (np.random.choice(websites), np.random.choice(websites))
        for _ in range(num_processes)
    ]

    # Use multiprocessing to generate new websites from pairs
    with multiprocessing.Pool(num_processes) as pool:
      new_websites = pool.starmap(generate_website, pairs)

    new_websites = [website for sublist in new_websites for website in sublist]
    logging.info('Generated %d new websites.', len(new_websites))
    combined_websites.extend(new_websites)
    logging.info('Total websites: %d', len(combined_websites))

  logging.info(
      'All Workers Finished. Generated %d websites.', len(combined_websites)
  )
  with open(
      os.path.join(_OUTPUT_DIR.value, 'combined_websites.pkl'), 'wb'
  ) as f:
    pickle.dump(combined_websites, f)


if __name__ == '__main__':
  app.run(main)
