import numpy as np

from a2perf.domains.web_navigation.gwob.CoDE import web_primitives

PASSIVE_PRIMITIVES = [
    'carousel',
    'cart',
    'dealmedia',
    'deck',
    'footer1',
    'forgotpassowrd',
    'forgotusername',
    'header',
    'header_login',
    'header_select_items',
    'inpgroup1',
    'navbar',
    'next_checkout',
    'next_login',
    'next_login_page',
    'submit',
]
ACTIVE_PRIMITIVES = [
    'addressline1',
    'addressline2',
    'cabin',
    'captcha',
    'cc',
    'cccvv',
    'ccexpdate',
    'ccnumber',
    'city',
    'departureairport',
    'departuredate',
    'destinationairport',
    'destinationdate',
    'firstname',
    'flighttype',
    'fullname',
    'lastname',
    'numberofpeople',
    'password',
    'rememberme',
    'state',
    'stayloggedin',
    'username',
    'zipcode',
]

PRIMITIVE_TO_INTERACTABLE_ELEMENTS = {
    'addressline1': 1,
    'addressline2': 1,
    'cabin': 2,
    'captcha': 1,
    'carousel': 7,
    'cart': 4,  # TODO: check
    'cc': 2,  # credit_card_type
    'cccvv': 1,  # credit_card_verification_code
    'ccexpdate': 1,
    'ccnumber': 1,
    'city': 1,
    'dealmedia': 1,
    'deck': 8,
    'departureairport': 1,
    'departuredate': 1,
    'destinationairport': 1,
    'destinationdate': 1,
    'firstname': 1,
    'flighttype': 2,
    'footer1': 4,
    'forgotpassowrd': 1,
    'forgotusername': 1,
    'fullname': 1,
    'header': 1,
    'header_login': 1,
    'header_select_items': 1,
    'inpgroup1': 1,
    'lastname': 1,
    'navbar': 6,
    'next_checkout': 1,
    'next_login': 1,
    'next_login_page': 1,
    'numberofpeople': 1,
    'password': 1,
    'rememberme': 1,
    'state': 2,
    'stayloggedin': 1,
    'submit': 1,
    'username': 1,
    'zipcode': 1,
}


def concept_id_from_concept_name(concept_name):
  """Returns the concept id for a given concept name."""
  return web_primitives.CONCEPTS.index(concept_name)


class Primitive(object):

  def __init__(
      self, name, primitive_id, num_interactable_elements, is_active=False
  ):
    self.name = name
    self.primitive_id = primitive_id
    self.is_active = is_active
    self.bonus_difficulty = None  # TODO: Add bonus difficulty for active primitives that need text input
    self.num_interactable_elements = num_interactable_elements

  def __str__(self):
    return f'{self.name}:{self.primitive_id}. Active: {self.is_active}'

  def __repr__(self):
    return self.__str__()


class Website(object):
  """A website is a sequence of pages."""

  def __init__(self, design=None, first_page=None, ):
    self._pages = []
    self.difficulty = 0
    self._num_possible_correct_steps = 0
    self._total_num_primitives = 0

    assert (first_page is None) != (
        design is None), 'Either first_page or design must be provided, but not both.'

    if design is not None:
      self._pages = []
      for page_id in range(design['number_of_pages']):
        page = Page(page_id)
        for primitive_id, primitive_name in zip(
            design['action_page'], design['action']
        ):
          if primitive_id == page_id:
            primitive = generate_primitive(
                web_primitives.CONCEPTS[primitive_name], primitive_name
            )
            page.add_primitive(primitive)
        self.add_page(page)
      self.update()

    if first_page is not None:
      self._pages = [first_page]
      self.difficulty = first_page.difficulty
      self._num_possible_correct_steps = first_page.num_possible_correct_interactions
      self._total_num_primitives = len(first_page.primitives)

  @property
  def num_pages(self):
    return len(self._pages)

  def update(self):
    """Updates state of the website."""
    self.difficulty = 0
    self._num_possible_correct_steps = 0
    self._total_num_primitives = 0
    for page in self._pages:
      self.difficulty += page.difficulty
      self._num_possible_correct_steps += page.num_possible_correct_interactions
      self._total_num_primitives += len(page.primitives)

  def add_page(self, page):
    """Adds a page to the website."""

    self.difficulty += page.difficulty
    self._pages.append(page)
    self._num_possible_correct_steps += page.num_possible_correct_interactions
    self._total_num_primitives += len(page.primitives)

  def to_design(self):
    """Returns a design dictionary for the website."""
    design = {
        'number_of_pages': self.num_pages,
        'action': [],
        'action_page': [],
    }
    for i, page in enumerate(self._pages):
      for primitive in page.primitives:
        design['action'].append(primitive.primitive_id)
        design['action_page'].append(i)
    return design


class Page(object):
  """A page is a collection of primitives."""

  def __init__(self, page_id, next_page=None):
    self.page_id = page_id
    self.next_page = next_page
    self.num_passive_primitives = 0
    self.num_active_primitives = 0
    self.num_interactable_elements = 0
    self.num_possible_correct_interactions = 0
    self.primitives = []
    self.difficulty = 0

  def __str__(self):
    return (
        f'Page {self.page_id}. Difficulty: {self.update_difficulty()}. Active'
        f' primitives: {self.num_active_primitives}. Passive primitives:'
        f' {self.num_passive_primitives} Next page:'
        f' {self.next_page.page_id if self.next_page is not None else None}'
    )

  def __repr__(self):
    return self.__str__()

  def add_primitive(self, primitive):
    """Adds a primitive to the page."""
    self.num_interactable_elements += primitive.num_interactable_elements
    if primitive.is_active:
      self.num_active_primitives += 1

      # Possible to interact with all fields of active primitives (for now)
      self.num_possible_correct_interactions += (
          primitive.num_interactable_elements
      )
    else:
      self.num_passive_primitives += 1

    self.primitives.append(primitive)
    self.update_difficulty()

  def update_difficulty(self):
    """Computes the difficulty of the page.

    The difficulty of a page is defined by the probability of a random agent
    interacting with the correct
    primitives.
    """
    probability_correct = (
        self.num_possible_correct_interactions
        / self.num_interactable_elements + 1e-5
    )
    self.difficulty = -np.log(probability_correct)
    return self.difficulty


def generate_primitive(primitive_name, primitive_id):
  """Generates a primitive."""
  is_active = primitive_name in ACTIVE_PRIMITIVES
  return Primitive(
      name=primitive_name,
      primitive_id=primitive_id,
      is_active=is_active,
      num_interactable_elements=PRIMITIVE_TO_INTERACTABLE_ELEMENTS[
        primitive_name
      ],
  )


if __name__ == '__main__':
  pass
