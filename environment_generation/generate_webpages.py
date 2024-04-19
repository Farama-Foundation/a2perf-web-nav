import json
import multiprocessing
import os

import numpy as np
from absl import app
from absl import flags
from absl import logging
import functools

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
  """
  Generates a single page with primitives.

  Determines whether to add more pages based on random primitive selection.

  Args:
      page_id (str): The ID of the page to generate.

  Returns:
      Tuple[website_util.Page, bool]: A tuple containing the generated page object
      and a boolean indicating whether to add more pages after this one.

  Raises:
      None

  Note:
      The function uses a loop to randomly select primitives to add to the page.
      It stops adding primitives when it encounters the "STOP_PRIMITIVE"
      or "NEXT_PAGE_PRIMITIVE" primitive.

  Example:
      To generate a page with ID "example_page_id", you can use:

      >>> page, make_next_page = generate_page("example_page_id")
  """
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


def generate_website(seed):
  """
  Generates a complete website by adding pages until a stopping condition is met.

  Args:
      seed (int): The seed value for the random number generator.

  Returns:
      website_util.Website: The generated website object containing all pages.

  Raises:
      None

  Note:
      The function iteratively generates pages using the `generate_page` function
      until a stopping condition is met. The stopping condition is determined
      by the `make_next_page` flag returned by the `generate_page` function.

  Example:
      To generate a website with a given seed value, you can use:

      >>> website = generate_website(123)
  """
  # We need to seed the random number generator for each process
  np.random.seed(seed)

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
  """
  Main function to orchestrate website generation process.

  Args:
      _: Unused. Placeholder for command-line arguments.

  Returns:
      None

  Raises:
      None

  Note:
      This function orchestrates the process of generating multiple websites
      in parallel. It generates unique seeds for each website, creates output
      directories as needed, and uses multiprocessing to generate websites
      concurrently.

  Example:
      To execute the main function, you can use:

      >>> main()
  """
  seeds = [_SEED.value + i for i in range(_NUM_WEBSITES.value)]
  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  # Parallel generation of websites, no arguments needed
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    websites = pool.starmap(generate_website,
                            zip(seeds))  # Generate websites in parallel

  logging.info('Generated %d websites.', len(websites))

  # Sort the websites by difficulty
  websites.sort(key=lambda website: website.difficulty)

  # Print out summary statistics for number of pages, difficulty, and number of primitives
  logging.info('Website statistics:')
  difficulties = [website.difficulty for website in websites]
  num_pages = [len(website._pages) for website in websites]
  num_primitives = [len(page.primitives) for website in websites for page in
                    website._pages]

  for values, name in zip([difficulties, num_pages, num_primitives],
                          ['difficulty', 'num_pages', 'num_primitives']):
    logging.info('  %s: mean=%f, std=%f, min=%f, max=%f', name, np.mean(values),
                 np.std(values), np.min(values), np.max(values))

    # Convert each website to its design representation in parallel
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    website_designs = pool.map(website_util.Website.convert_to_design, websites)
  logging.info('Converted %d websites to designs.', len(website_designs))

  # Website designs is a list of dictionaries. It does not make sense to save duplicate designs, so let's only save unique design dictionaries.

  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    json_dump_function = functools.partial(json.dumps, sort_keys=True)
    website_designs = pool.map(json_dump_function, website_designs)
    website_designs = list(set(website_designs))
    website_designs = pool.map(json.loads, website_designs)

  logging.info('Generated %d unique website designs.', len(website_designs))

  # Save websites to disk
  with open(os.path.join(_OUTPUT_DIR.value, 'website_designs.json'), 'w') as f:
    json.dump(website_designs, f)


if __name__ == '__main__':
  app.run(main)
