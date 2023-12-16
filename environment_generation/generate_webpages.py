import collections
from absl import logging
import os
import pickle
import copy
import numpy as np
from absl import app
from absl import flags

from a2perf.domains.web_navigation.environment_generation import website_util
from a2perf.domains.web_navigation.gwob.CoDE import web_primitives

flags.DEFINE_integer(
    'num_difficulty_levels', 10, 'Number of difficulty levels to generate.'
)

flags.DEFINE_integer(
    'seed', 1, 'Random seed for web page generation'
)
flags.DEFINE_multi_string(
    'difficulty_names',
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Names of difficulty levels.',
)
flags.DEFINE_integer('num_pages', 10000, 'Initial number of pages to generate.')
flags.DEFINE_integer(
    'max_num_primitives', 25, 'Maximum number of primitives per page.'
)
flags.DEFINE_integer(
    'max_horizon', 25, 'Maximum number of pages in a sequence.'
)
flags.DEFINE_integer(
    'websites_to_generate', 40000, 'Number of websites to generate.'
)
flags.DEFINE_string(
    'output_dir', './data', 'Directory to save generated websites to.'
)
FLAGS = flags.FLAGS


def estimate_num_websites_to_generate(num_primitives, avg_num_primitives_per_page, num_websites_to_generate_estimate):
    l_values = np.array(range(1, avg_num_primitives_per_page + 1))
    k_choose_two = num_primitives * (num_primitives - 1) / 2

    x = ((num_primitives - 2) / num_primitives) ** (avg_num_primitives_per_page - l_values) * (
            1 / num_primitives) ** l_values
    prob_j_never_appears_on_page_one_given_i_appears_on_page_one = 1 - np.sum(x)
    prob_j_appears_on_page_one_given_i_appears_on_page_one = 1 - prob_j_never_appears_on_page_one_given_i_appears_on_page_one
    prob_i_never_appears_on_page_one = ((num_primitives - 1) / num_primitives) ** avg_num_primitives_per_page
    prob_i_appears_on_page_one = 1 - prob_i_never_appears_on_page_one
    prob_both_ij_appear_on_page_one = prob_i_appears_on_page_one * prob_j_appears_on_page_one_given_i_appears_on_page_one
    prob_ij_never_appear_on_page_one = 1 - prob_both_ij_appear_on_page_one
    prob_ij_never_appear_on_any_page = prob_ij_never_appear_on_page_one ** num_websites_to_generate_estimate
    prob_ij_appear_on_some_page_atleast_once = 1 - prob_ij_never_appear_on_any_page

    prob_each_pair_of_prims_on_some_page_atleast_once = prob_ij_appear_on_some_page_atleast_once ** k_choose_two
    return prob_each_pair_of_prims_on_some_page_atleast_once


def main(_):
    assert (
            FLAGS.websites_to_generate >= FLAGS.num_pages
    ), 'Number of pages must be less than number of websites to generate.'

    # Set random seed
    np.random.seed(FLAGS.seed)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if os.path.exists(os.path.join(FLAGS.output_dir, 'pages.pkl')):
        with open(os.path.join(FLAGS.output_dir, 'pages.pkl'), 'rb') as f:
            pages = pickle.load(f)
        logging.info(f'Loaded {len(pages)} pages from {FLAGS.output_dir}.')
    else:
        # Randomly generate single web pages
        pages = []
        for page_id in range(FLAGS.num_pages):
            page = website_util.Page(page_id)
            num_primitives = np.random.randint(1, FLAGS.max_num_primitives + 1)
            for _ in range(num_primitives):
                primitive_name = np.random.choice(web_primitives.CONCEPTS[1:])
                primitive_id = web_primitives.CONCEPTS.index(primitive_name)
                primitive = website_util.generate_primitive(primitive_name, primitive_id)

                logging.info(f'Adding primitive {primitive} to page {page_id}.')
                page.add_primitive(primitive)

                if page.num_possible_correct_interactions >= FLAGS.max_horizon:
                    break

            pages.append(page)
        logging.info(f'Generated {len(pages)} pages.')

        # Save the initial set of generated pages
        with open(os.path.join(FLAGS.output_dir, 'pages.pkl'), 'wb') as f:
            pickle.dump(pages, f)

    # To create websites, randomly connect the pages together
    websites = [website_util.Website(p) for p in pages]
    generated_websites = len(websites)

    while generated_websites < FLAGS.websites_to_generate:
        website = np.random.choice(websites)
        new_website = website_util.Website(first_page=website._pages[0])
        new_website._pages = copy.copy(website._pages)

        # Randomly select a next page from websites
        next_website = np.random.choice(websites)
        new_website._pages.extend(next_website._pages)
        new_website.update()
        assert new_website._num_possible_correct_steps == (
                website._num_possible_correct_steps + next_website._num_possible_correct_steps
        ), 'Number of possible correct steps should be equal to sum of number of possible correct steps of the two websites.'
        if new_website._num_possible_correct_steps <= FLAGS.max_horizon:
            websites.append(new_website)
            logging.info(f'Generated website {len(websites)}.')
            generated_websites += 1
        # Save the generated websites every 10k websites
        if generated_websites % 10000 == 0:
            with open(os.path.join(FLAGS.output_dir, f'websites_{generated_websites}.pkl'), 'wb') as f:
                pickle.dump(websites, f)
            logging.info(f'Saved {len(websites)} websites.')
    logging.info(f'Generated {len(websites)} websites.')

    # Sort websites by difficulty
    websites.sort(key=lambda x: x.difficulty)
    logging.info('Sorted websites by difficulty.')

    # Split websites into difficulty levels
    partitioned_websites = np.array_split(websites, FLAGS.num_difficulty_levels)

    # Convert websites into design dictionaries and save both designs and websites
    designs = collections.defaultdict(list)
    for difficulty_num, websites in enumerate(partitioned_websites):
        for website in websites:
            designs[FLAGS.difficulty_names[difficulty_num]].append(website.to_design())

        # Save designs to files with numpy
        # Save website objects to files with pickle
        with open(f'{FLAGS.output_dir}/websites_{FLAGS.difficulty_names[difficulty_num]}.pkl', 'wb') as f:
            pickle.dump(websites, f)

    logging.info('Converted and saved websites and designs.')


if __name__ == '__main__':
    app.run(main)
