import os
import collections
import copy
import logging

import numpy as np
from rl_perf.domains.web_nav.CoDE import web_primitives
from absl import app
from absl import flags

import website_util

flags.DEFINE_integer('num_difficulty_levels', 6, 'Number of difficulty levels to generate.')
flags.DEFINE_multi_string('difficulty_names', ['1', '2', '3', '4', '5', '6'], 'Names of difficulty levels.')
flags.DEFINE_integer('num_pages', 1000, 'Initial number of pages to generate.')
flags.DEFINE_integer('max_num_primitives', 40, 'Maximum number of primitives per page.')
flags.DEFINE_integer('max_horizon', 10, 'Maximum number of pages in a sequence.')
flags.DEFINE_integer('websites_to_generate', 10000, 'Number of websites to generate.')
flags.DEFINE_string('output_dir', './data', 'Directory to save generated websites to.')
FLAGS = flags.FLAGS


def main(_):
    assert FLAGS.websites_to_generate >= FLAGS.num_pages, 'Number of pages must be less than number of websites to ' \
                                                          'generate.'

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Randomly generate single web pages
    pages = []
    for page_id in range(FLAGS.num_pages):
        page = website_util.Page(page_id)
        num_primitives = np.random.randint(1, FLAGS.max_num_primitives + 1)
        for _ in range(num_primitives):
            primitive_name = np.random.choice(web_primitives.CONCEPTS[1:])
            primitive_id = web_primitives.CONCEPTS.index(primitive_name)
            primitive = website_util.generate_primitive(primitive_name, primitive_id)
            page.add_primitive(primitive)
        pages.append(page)
    logging.info(f'Generated {len(pages)} pages.')

    # To create websites, randomly connect the pages together
    websites = [website_util.Website(p) for p in pages]
    for _ in range(FLAGS.websites_to_generate - len(websites)):
        website = np.random.choice(websites)
        new_website = copy.deepcopy(website)

        # Randomly select a next page from websites
        next_website = np.random.choice(websites)
        new_website.add_page(copy.deepcopy(next_website.first_page))

        websites.append(new_website)
    logging.info(f'Generated {len(websites)} websites.')

    # Sort websites by difficulty
    websites.sort(key=lambda x: x.difficulty)
    logging.info('Sorted websites by difficulty.')

    # Split websites into difficulty levels
    partitioned_websites = np.array_split(websites, FLAGS.num_difficulty_levels)

    # Convert websites into design dictionaries
    designs = collections.defaultdict(list)
    for difficulty_num, websites in enumerate(partitioned_websites):
        for website in websites:
            designs[FLAGS.difficulty_names[difficulty_num]].append(website.to_design())
    logging.info('Converted websites to designs.')

    # Save designs to files with numpy
    for difficulty_name, design in designs.items():
        np.save(f'{FLAGS.output_dir}/design_{difficulty_name}.npy', design)


if __name__ == '__main__':
    app.run(main)
