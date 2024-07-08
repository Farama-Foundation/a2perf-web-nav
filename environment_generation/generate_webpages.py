import functools
import json
import multiprocessing
import os
import zipfile

import numpy as np
from absl import app
from absl import flags
from absl import logging

from a2perf.domains.web_navigation.environment_generation import website_util
from a2perf.domains.web_navigation.gwob.CoDE import web_primitives

# Define flags for script configuration
_SEED = flags.DEFINE_integer('seed', 1234,
                             'Random seed for web page generation')
_NUM_PROCESSES = flags.DEFINE_integer('num_processes', None,
                                      'Number of processes to use.')
_NUM_WEBSITES = flags.DEFINE_integer('num_websites', 100000,
                                     'Number of websites to generate.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', './data',
                                  'Directory to save generated websites to.')
_NUM_DIFFICULTY_LEVEL_BINS = flags.DEFINE_list('num_difficulty_level_bins',
                                               [0.0, 0.25, 0.5, 0.75],
                                               'List of probabilities for each difficulty level bin.')
_MAX_NUM_WEBSITES_PER_DIFFICULTY_LEVEL = flags.DEFINE_integer(
    'max_num_websites_per_difficulty_level', 100,
    'Maximum number of websites to generate per difficulty level.')

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


def split_websites_into_difficulty_levels(websites, website_designs,
    difficulty_bins, max_num_websites_per_difficulty_level=None):
  """Websites are split into difficulty levels based on the probability that
  a random agent will reach the end of the website."""

  num_levels = len(difficulty_bins)
  difficulty_scores = np.array([website.difficulty for website in websites])
  probabilities = np.exp(-difficulty_scores)

  difficulty_levels = np.digitize(probabilities, difficulty_bins)

  # Split the websites into difficulty levels. num_levels -1 accounts for the
  # extra bin for websites above the lowest probability threshold
  website_designs_by_difficulty = [[] for _ in range(num_levels)]
  websites_by_difficulty = [[] for _ in range(num_levels)]
  for i, (website, website_design) in enumerate(zip(websites, website_designs)):
    difficulty_level = difficulty_levels[
                         i] - 1  # Difficulty levels are 1-indexed
    website_designs_by_difficulty[difficulty_level].append(website_design)
    websites_by_difficulty[difficulty_level].append(website)

  # For each level, get statistics on the number of pages, difficulty, and number of primitives
  for i, websites in enumerate(websites_by_difficulty):
    difficulties = [website.difficulty for website in websites]
    num_pages = [len(website._pages) for website in websites]
    num_primitives = [len(page.primitives) for website in websites for page in
                      website._pages]

    if len(websites) == 0:
      logging.warning('No websites in difficulty level %d.', i)
      continue

    logging.info('Difficulty level %d:', i)
    for values, name in zip([difficulties, num_pages, num_primitives],
                            ['difficulty', 'num_pages', 'num_primitives']):
      logging.info('  %s: mean=%f, median=%f, std=%f, min=%f, max=%f',
                   name, np.mean(values), np.median(values), np.std(values),
                   np.min(values), np.max(values))

  # We might have an extra bin for websites above the lowest probability threshold
  websites_by_difficulty = websites_by_difficulty[:num_levels - 1]
  website_designs_by_difficulty = website_designs_by_difficulty[:num_levels - 1]

  # Easier website should have lower index
  website_designs_by_difficulty = website_designs_by_difficulty[::-1]

  # Limit the number of websites per difficulty level
  if max_num_websites_per_difficulty_level is not None:
    for i in range(len(website_designs_by_difficulty)):
      if len(website_designs_by_difficulty[
               i]) > max_num_websites_per_difficulty_level:
        website_designs_by_difficulty[i] = np.random.choice(
            website_designs_by_difficulty[i],
            max_num_websites_per_difficulty_level, replace=False).tolist()
  return website_designs_by_difficulty


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
    pool.close()
    pool.join()

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
    logging.info('  %s: mean=%f, median=%f, std=%f, min=%f, max=%f',
                 name, np.mean(values), np.median(values), np.std(values),
                 np.min(values), np.max(values))

    # Convert each website to its design representation in parallel
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    website_designs = pool.map(website_util.Website.convert_to_design, websites)
    pool.close()
    pool.join()
  logging.info('Converted %d websites to designs.', len(website_designs))

  # Save only the unique website designs
  with multiprocessing.Pool(_NUM_PROCESSES.value) as pool:
    json_dump_function = functools.partial(json.dumps, sort_keys=True)
    website_designs = pool.map(json_dump_function, website_designs)
    website_designs = list(set(website_designs))
    website_designs = pool.map(json.loads, website_designs)
    pool.close()
    pool.join()

  logging.info('Generated %d unique website designs.', len(website_designs))

  # Save websites to disk
  website_design_path = os.path.join(_OUTPUT_DIR.value, 'website_designs.json')
  with open(website_design_path, 'w') as f:
    json.dump(website_designs, f)

  difficulty_levels = split_websites_into_difficulty_levels(websites,
                                                            website_designs,
                                                            difficulty_bins=_NUM_DIFFICULTY_LEVEL_BINS.value,
                                                            max_num_websites_per_difficulty_level=_MAX_NUM_WEBSITES_PER_DIFFICULTY_LEVEL.value)

  # Save the difficulty levels to disk as JSON
  difficulty_levels_path = os.path.join(_OUTPUT_DIR.value, 'difficulty_levels')
  for i, difficulty_level in enumerate(difficulty_levels):
    difficulty_level_path = os.path.join(difficulty_levels_path,
                                         f'{i:02d}.json')
    os.makedirs(os.path.dirname(difficulty_level_path), exist_ok=True)
    with open(difficulty_level_path, 'w') as f:
      json.dump(difficulty_level, f)

  # Zip the difficulty levels in a single zip file
  with zipfile.ZipFile(difficulty_levels_path + '.zip', 'w') as zipf:
    for i in range(len(difficulty_levels)):
      zipf.write(os.path.join(difficulty_levels_path, f'{i:02d}.json'),
                 f'{i:02d}.json')


if __name__ == '__main__':
  app.run(main)
