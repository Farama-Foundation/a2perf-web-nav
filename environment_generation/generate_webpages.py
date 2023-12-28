import json
import multiprocessing
import os

import numpy as np
from absl import app
from absl import flags
from absl import logging

from a2perf.domains.web_navigation.environment_generation import website_util
from a2perf.domains.web_navigation.gwob.CoDE import web_primitives

# Define flags for script configuration
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed for web page generation')
_NUM_PROCESSES = flags.DEFINE_integer('num_processes', None,
                                      'Number of processes to use.')
_NUM_WEBSITES = flags.DEFINE_integer('num_websites', 10000,
                                     'Number of websites to generate.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', './data',
                                  'Directory to save generated websites to.')

# Extending the original web primitives with a custom primitive for signaling new page creation
CUSTOM_CONCEPTS = web_primitives.CONCEPTS + ['#next_page#']
STOP_PRIMITIVE = '#none#'
NEXT_PAGE_PRIMITIVE = '#next_page#'


def generate_page(page_id):
  """Generates a single page with primitives. Determines whether to add more pages based on random primitive selection."""
  page = website_util.Page(page_id)
  make_next_page = False

  while True:
    primitive = np.random.choice(CUSTOM_CONCEPTS)
    if primitive == NEXT_PAGE_PRIMITIVE:
      make_next_page = True
      break  # Indicates that a new page should be added after this one
    elif primitive == STOP_PRIMITIVE:
      break  # Stops generating more primitives for this page
    else:
      primitive_obj = website_util.Primitive(
          name=primitive)  # Create primitive object
      page.add_primitive(primitive_obj)

  return page, make_next_page


def generate_website():
  """Generates a complete website by adding pages until a stopping condition is met."""
  pages = []
  current_page_id = 0
  while True:
    page, make_next_page = generate_page(current_page_id)
    pages.append(page)
    if not make_next_page:
      break  # Stop generating more pages
    current_page_id += 1

  return website_util.Website(pages=pages)


def main(_):
  """Main function to orchestrate website generation process."""
  np.random.seed(_SEED.value)
  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  # Parallel generation of websites, no arguments needed
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    websites = pool.starmap(generate_website,
                            [() for _ in range(_NUM_WEBSITES.value)])
  logging.info('Generated %d websites.', len(websites))

  # Sort the websites by difficulty
  websites.sort(key=lambda website: website.difficulty)

  # Convert each website to its design representation in parallel
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    website_designs = pool.map(website_util.Website.convert_to_design, websites)
  logging.info('Converted %d websites to designs.', len(website_designs))

  # Website designs is a list of dictionaries. It does not make sense to save duplicate designs, so let's only save unique design dictionaries.
  website_designs = list(
      {json.dumps(d, sort_keys=True) for d in website_designs})
  website_designs = [json.loads(d) for d in website_designs]

  logging.info('Generated %d unique website designs.', len(website_designs))

  # Save websites to disk
  with open(os.path.join(_OUTPUT_DIR.value, 'website_designs.json'), 'w') as f:
    json.dump(website_designs, f)


if __name__ == '__main__':
  app.run(main)
