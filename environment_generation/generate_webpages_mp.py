import copy
import multiprocessing
import os
import pickle

import numpy as np
from absl import app
from absl import flags
from absl import logging

from a2perf.domains.web_navigation.environment_generation import website_util
from a2perf.domains.web_navigation.gwob.CoDE import web_primitives

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
_NUM_WEBSITES_PER_WORKER = flags.DEFINE_integer(
    'num_websites_per_worker', 1000,
    'Number of websites to generate per worker.'
)


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


def generate_websites(website_designs, num_websites_to_generate, max_horizon):
  """Generates new websites from existing websites."""

  new_websites = []
  while len(new_websites) < num_websites_to_generate:
    current_website = np.random.choice(website_designs)
    next_website = np.random.choice(website_designs)

    current_website = website_util.Website(design=current_website)
    next_website = website_util.Website(design=next_website)

    # We create a new website object, then copy over the pages from the two
    new_website = website_util.Website(first_page=current_website._pages[0])
    new_website._pages = copy.copy(current_website._pages)
    new_website._pages.extend(next_website._pages)
    new_website.update()

    assert new_website._num_possible_correct_steps == (
        current_website._num_possible_correct_steps
        + next_website._num_possible_correct_steps
    ), 'Number of possible correct steps should be the sum of the two websites.'

    del current_website
    del next_website

    if new_website._num_possible_correct_steps <= max_horizon:
      new_websites.append(new_website)

  # Return designs since they are much easier to pickle
  new_websites = [website.to_design() for website in new_websites]
  return new_websites


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
  websites = [website_util.Website(first_page=p).to_design() for p in pages]

  # Generate new websites
  while len(websites) < _WEBSITES_TO_GENERATE.value:
    with multiprocessing.Pool(num_processes) as pool:
      new_websites = pool.starmap(generate_websites, [
          (websites, _NUM_WEBSITES_PER_WORKER.value,
           _MAX_HORIZON.value)] * num_processes)

    new_websites = [website for sublist in new_websites for website in sublist]
    logging.info('Generated %d new websites.', len(new_websites))
    websites.extend(new_websites)

    logging.info('Total websites: %d', len(websites))

  logging.info(
      'All Workers Finished. Generated %d websites.', len(websites)
  )
  websites = [website_util.Website(design=design) for design in websites]

  # Sort the websites by difficulty
  websites.sort(key=lambda website: website.difficulty)
  logging.info('Sorted websites by difficulty.')

  # Convert websites to designs
  designs = [website.to_design() for website in websites]
  with open(os.path.join(_OUTPUT_DIR.value, 'designs.pkl'), 'wb') as f:
    pickle.dump(designs, f)
    logging.info('Saved designs to %s', _OUTPUT_DIR.value)


if __name__ == '__main__':
  app.run(main)
