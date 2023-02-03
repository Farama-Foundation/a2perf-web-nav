import collections

import gin
import numpy as np
from CoDE import vocabulary_node
from CoDE import web_environment
from CoDE import web_primitives

PASSIVE_PRIMITIVES = ['carousel', 'cart', 'dealmedia', 'deck', 'footer1', 'forgotpassowrd', 'forgotusername', 'header',
                      'header_login', 'header_select_items', 'inpgroup1', 'navbar', 'next_checkout',
                      'next_login',
                      'next_login_page', 'submit']
ACTIVE_PRIMITIVES = ['addressline1', 'addressline2', 'cabin', 'captcha', 'cc', 'cccvv', 'ccexpdate', 'ccnumber', 'city',
                     'departureairport', 'departuredate', 'destinationairport', 'destinationdate', 'firstname',
                     'flighttype', 'fullname', 'lastname', 'numberofpeople', 'password', 'rememberme', 'state',
                     'stayloggedin', 'username', 'zipcode']

PRIMITIVE_TO_INTERACTABLE_ELEMENTS = {'addressline1': 1,
                                      'addressline2': 1,
                                      'cabin': 1,
                                      'captcha': 1,
                                      'cc': 1,
                                      'cccvv': 1,
                                      'ccexpdate': 1,
                                      'ccnumber': 1,
                                      'city': 1,
                                      'departureairport': 1,
                                      'departuredate': 1,
                                      'destinationairport': 1,
                                      'destinationdate': 1,
                                      'firstname': 1,
                                      'flighttype': 1,
                                      'fullname': 1,
                                      'lastname': 1,
                                      'numberofpeople': 1,
                                      'password': 1,
                                      'rememberme': 1,
                                      'state': 1,
                                      'stayloggedin': 1,
                                      'username': 1,
                                      'zipcode': 1,
                                      'carousel': 1,
                                      'cart': 1,
                                      'dealmedia': 1,
                                      'deck': 1,
                                      'footer1': 1,
                                      'forgotpassowrd': 1,
                                      'forgotusername': 1,
                                      'header': 1,
                                      'header_login': 1,
                                      'header_select_items': 1,
                                      'inpgroup1': 1,
                                      'navbar': 1,
                                      'next_checkout': 1,
                                      'next_login': 1,
                                      'next_login_page': 1,
                                      'submit': 1}

MAX_PRIMITIVES_PER_PAGE = 40


class Primitive(object):
    def __init__(self, name, is_active=False):
        self.name = name
        self.is_active = is_active
        self.bonus_difficulty = None  # TODO: Add bonus difficulty for active primitives that need text input
        self.num_interactable_elements = PRIMITIVE_TO_INTERACTABLE_ELEMENTS[name]


class Page(object):
    def __init__(self, page_id, next_page=None):
        self.page_id = page_id
        self.next_page = next_page
        self.num_passive_primitives = 0
        self.num_active_primitives = 0
        self.num_interactable_elements = 0
        self.primitives = []
        self.difficulty = None

    def add_primitive(self, primitive):
        """Adds a primitive to the page."""
        if primitive.is_active:
            self.num_active_primitives += 1
        else:
            self.num_passive_primitives += 1

        self.num_interactable_elements += primitive.num_interactable_elements
        self.primitives.append(primitive)

    def compute_difficulty(self):
        """Computes the difficulty of the page.

        The difficulty of a page is defined by the probability of a random agent interacting with the correct
        primitives.
        """
        if self.difficulty is None:
            num_possible_correct_interactions = 1 if self.num_active_primitives == 0 else self.num_active_primitives
            probability_correct = num_possible_correct_interactions / self.num_interactable_elements
            self.difficulty = -np.log(probability_correct)
        return self.difficulty

    def _compute_sequence_difficulty(self):
        """Computes the difficulty of the sequence of pages starting at this page."""
        next_difficulty = 0 if self.next_page is None else self.next_page.compute_difficulty()
        return self.compute_difficulty() + next_difficulty


def generate_primitive(primitive_name):
    is_active = primitive_name in ACTIVE_PRIMITIVES
    return Primitive(primitive_name, is_active)


def generate_pages(number_of_pages):
    pass


if __name__ == '__main__':
    gin.parse_config_files_and_bindings(["/web_nav/gwob/configs/envdesign.gin"], None)

    numbers = range(0, 41)

    d2 = {'number_of_pages': len(numbers),
          'action_page': numbers, 'action': numbers, }

    env = web_environment.GMiniWoBWebEnvironment(
            base_url="file:///web_nav/gwob/",
            global_vocabulary=vocabulary_node.LockedVocabulary())
    env.design_environment(env_design=d2, auto_num_pages=True)

    d = {
            'number_of_pages': 1,
            'action_page': [0, 0, 0, 0],
            'action':
                [web_primitives.CONCEPTS.index(x) for x in ('deck', 'deck', 'deck', 'deck')]}
