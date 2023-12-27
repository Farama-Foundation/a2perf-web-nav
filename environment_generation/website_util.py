import numpy as np
from absl import app

from a2perf.domains.web_navigation.gwob.CoDE import web_primitives

# NOTE: Generated using a2perf.domains.web_navigation.gwob.CoDE
# .web_primitives.generate_primitive_info_dict()
PRIMITIVE_INFO_DICT = {
    "navbar": {
        "active": False,
        "controls": {
            "menuItems": [
                "Home",
                "Login",
                "Account",
                "Cart",
                "Checkout"
            ],
            "endOnClick": True
        },
        "num_dom_elements": 10
    },
    "carousel": {
        "active": False,
        "controls": {
            "numItems": 5,
            "itemNames": [
                "1",
                "2",
                "3",
                "4",
                "5"
            ],
            "endOnClick": True
        },
        "num_dom_elements": 13
    },
    "dealmedia": {
        "active": False,
        "controls": {
            "title": "Deal of the Day",
            "text": "Gaming workstation",
            "link": "Get it today!",
            "endOnClick": True
        },
        "num_dom_elements": 12
    },
    "header_select_items": {
        "active": False,
        "controls": {
            "headerType": 5,
            "headerText": "Select items",
            "isCardHeader": False
        },
        "num_dom_elements": 19
    },
    "deck": {
        "active": False,
        "controls": {
            "numCards": 4,
            "cardTitles": [
                "Title 1",
                "Title 2"
            ],
            "cardText": [
                "Product description 1",
                "Product description 2"
            ],
            "cardNames": [
                "Card 1",
                "Card 2"
            ],
            "cardHeaders": [
                "$0.99",
                "$1.99"
            ],
            "numStars": [
                4,
                3
            ],
            "endOnClick": True
        },
        "num_dom_elements": 13
    },
    "next_login_page": {
        "active": False,
        "controls": {
            "buttonText": "Login"
        },
        "num_dom_elements": 20
    },
    "header_login": {
        "active": False,
        "controls": {
            "headerType": 5,
            "headerText": "Login",
            "isCardHeader": False
        },
        "num_dom_elements": 47
    },
    "username": {
        "active": True,
        "controls": {
            "putPlaceholder": True,
            "putLabel": False,
            "labelText": "Username"
        },
        "num_dom_elements": 19
    },
    "password": {
        "active": True,
        "controls": {
            "putPlaceholder": True,
            "putLabel": False,
            "labelText": "Password"
        },
        "num_dom_elements": 13
    },
    "rememberme": {
        "active": True,
        "controls": {
            "putLabel": True,
            "labelText": "Remember me"
        },
        "num_dom_elements": 13
    },
    "captcha": {
        "active": True,
        "controls": {
            "putLabel": True,
            "labelText": "Enter Captcha"
        },
        "num_dom_elements": 13
    },
    "stayloggedin": {
        "active": True,
        "controls": {
            "putLabel": True,
            "labelText": "Stay logged in"
        },
        "num_dom_elements": 13
    },
    "forgotusername": {
        "active": False,
        "controls": {
            "text": "Forgot user name.",
            "endOnClick": True
        },
        "num_dom_elements": 16
    },
    "forgotpassowrd": {
        "active": False,
        "controls": {
            "text": "Forgot password.",
            "endOnClick": True
        },
        "num_dom_elements": 67
    },
    "next_login": {
        "active": False,
        "controls": {
            "buttonText": "Login and Checkout"
        },
        "num_dom_elements": 13
    },
    "cart": {
        "active": False,
        "controls": {
            "wrapInCard": True,
            "numItems": 3,
            "itemNames": [
                "Shoe",
                "Bag",
                "Tshirt"
            ],
            "endOnClick": True
        },
        "num_dom_elements": 13
    },
    "next_checkout": {
        "active": False,
        "controls": {
            "buttonText": "Checkout"
        },
        "num_dom_elements": 13
    },
    "header": {
        "active": False,
        "controls": {
            "headerType": 5,
            "headerText": "Shipping Information",
            "isCardHeader": False
        },
        "num_dom_elements": 13
    },
    "firstname": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "First Name"
        },
        "num_dom_elements": 13
    },
    "lastname": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Last Name"
        },
        "num_dom_elements": 18
    },
    "addressline1": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Address"
        },
        "num_dom_elements": 21
    },
    "addressline2": {
        "active": True,
        "controls": {
            "putPlaceholder": True,
            "putLabel": False,
            "labelText": ""
        },
        "num_dom_elements": 12
    },
    "city": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "City"
        },
        "num_dom_elements": 12
    },
    "zipcode": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "ZIP Code"
        },
        "num_dom_elements": 13
    },
    "state": {
        "active": True,
        "controls": {
            "putLabel": False,
            "labelText": "State",
            "values": [
                "CA",
                "NY"
            ]
        },
        "num_dom_elements": 12
    },
    "submit": {
        "active": False,
        "controls": {
            "buttonText": "Place Order"
        },
        "num_dom_elements": 12
    },
    "footer1": {
        "active": False,
        "controls": {
            "footerItems": [
                "Contact",
                "Terms",
                "Support",
                "Full Site"
            ],
            "endOnClick": True
        },
        "num_dom_elements": 12
    },
    "inpgroup1": {
        "active": False,
        "controls": {
            "putPlaceholder": True,
            "putLabel": False,
            "labelText": "Search"
        },
        "num_dom_elements": 12
    },
    "cc": {
        "active": True,
        "controls": {
            "header": "Payment",
            "items": [
                "Credit Card",
                "Debit Card"
            ]
        },
        "num_dom_elements": 17
    },
    "fullname": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Full Name"
        },
        "num_dom_elements": 12
    },
    "ccnumber": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Credit card number"
        },
        "num_dom_elements": 12
    },
    "ccexpdate": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Expiration date"
        },
        "num_dom_elements": 12
    },
    "cccvv": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "CVV"
        },
        "num_dom_elements": 13
    },
    "departureairport": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "From"
        },
        "num_dom_elements": 12
    },
    "destinationairport": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "To"
        },
        "num_dom_elements": 13
    },
    "departuredate": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Depart"
        },
        "num_dom_elements": 12
    },
    "destinationdate": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Return"
        },
        "num_dom_elements": 13
    },
    "numberofpeople": {
        "active": True,
        "controls": {
            "putPlaceholder": False,
            "putLabel": True,
            "labelText": "Number of passengers"
        },
        "num_dom_elements": 10
    },
    "cabin": {
        "active": True,
        "controls": {
            "name": "cabin",
            "header": "Cabin",
            "items": [
                "Economy",
                "First"
            ]
        },
        "num_dom_elements": 12
    },
    "flighttype": {
        "active": True,
        "controls": {
            "name": "flighttype",
            "header": "",
            "items": [
                "Oneway",
                "Roundtrip"
            ]
        },
        "num_dom_elements": 13
    }
}


class Primitive(object):
  def __init__(self, name, primitive_id=None):
    """
    Initializes a primitive.

    Args:
      name (str): Name of the primitive.
      primitive_id (int): Id of the primitive, used to identify the primitive
                          especially when multiple primitives of the same type exist on a page.
    """
    self.name = name
    self.primitive_id = primitive_id
    self.num_dom_elements = PRIMITIVE_INFO_DICT[name]['num_dom_elements']
    self.is_active = PRIMITIVE_INFO_DICT[name]['active']

  def __str__(self):
    return f'{self.name}:{self.primitive_id}. Active: {self.is_active}'

  def __repr__(self):
    return self.__str__()


class Page(object):
  """
  A Page represents a collection of primitives within a website.
  """

  def __init__(self, page_id):
    self.primitives = []
    self.total_num_dom_elements = 0
    self.difficulty = 0
    self.page_id = page_id

  @property
  def num_primitives(self):
    """Returns the number of primitives on the page."""
    return len(self.primitives)

  @property
  def active_primitives(self):
    """Returns a list of active primitives on the page."""
    return [primitive for primitive in self.primitives if primitive.is_active]

  @property
  def passive_primitives(self):
    """Returns a list of passive (non-active) primitives on the page."""
    return [primitive for primitive in self.primitives if
            not primitive.is_active]

  def add_primitive(self, primitive):
    """
    Adds a primitive to the page.

    Args:
      primitive (Primitive): The primitive to be added to the page.
    """
    if primitive.name not in PRIMITIVE_INFO_DICT:
      raise ValueError(f"Primitive name '{primitive.name}' is not defined.")
    self.primitives.append(primitive)
    self.total_num_dom_elements += primitive.num_dom_elements

  def calculate_difficulty(self):
    """
    Computes and returns the difficulty of the page based on the probability
    of a random agent interacting with the correct primitives.

    Returns:
      float: The calculated difficulty of the page.
    """
    prob_all_active_primitives_match = 1 / np.math.factorial(
        len(self.active_primitives))
    prob_distracting_primitive_clicked = sum(
        [primitive.num_dom_elements for primitive in self.passive_primitives]
    ) / self.total_num_dom_elements

    prob_no_distracting_primitives_clicked = 1 - prob_distracting_primitive_clicked

    prob_website_filled_out_correctly = (
        prob_all_active_primitives_match * prob_no_distracting_primitives_clicked
    )
    self.difficulty = -np.log(prob_website_filled_out_correctly + 1e-8)
    return self.difficulty


class Website(object):
  """
  A Website is a sequence of pages, representing a complete website structure.
  """

  def __init__(self, design=None, pages=None):
    if not (design or pages):
      raise ValueError('Either design or pages must be provided.')
    if design is not None and pages is not None:
      raise ValueError('Either design or pages must be provided, but not both.')

    self._pages = []
    if pages is not None:
      self._pages = pages
    else:
      self._pages = [Page(page_id) for page_id in
                     range(design['number_of_pages'])]
      for i, (primitive_page_index, primitive_identifier) in enumerate(zip(
          design['action_page'], design['action']
      )):
        primitive_name = web_primitives.CONCEPTS[primitive_identifier]
        if primitive_name not in web_primitives.CONCEPTS:
          raise ValueError(
              f"Invalid primitive identifier: {primitive_identifier}")
        self._pages[primitive_page_index].add_primitive(
            Primitive(primitive_id=i, name=primitive_name))

    self._calculate_difficulty()

  def _calculate_difficulty(self):
    """
    Computes and updates the difficulty of the entire website.
    """
    self.difficulty = 0
    for page in self._pages:
      self.difficulty += page.calculate_difficulty()
    return self.difficulty

  def add_page(self, page):
    """
    Adds a page to the website.

    Args:
      page (Page): The page to be added to the website.
    """
    self._pages.append(page)
    self._calculate_difficulty()

  def remove_page(self, page):
    """
    Removes a page from the website.

    Args:
      page (Page): The page to be removed from the website.
    """
    self._pages.remove(page)
    self._calculate_difficulty()

  def convert_to_design(self):
    """
    Converts the website structure into a design dictionary format.

    Returns:
      dict: A dictionary representing the design of the website.
    """
    design = {
        'number_of_pages': len(self._pages),
        'action': [],
        'action_page': [],
    }
    for i, page in enumerate(self._pages):
      for primitive in page.primitives:
        design['action'].append(primitive.name)
        design['action_page'].append(i)
    return design


def main(_):
  import gymnasium as gym

  env = gym.make(id='WebNavigation-v0', difficulty=1,
                 browser_args=dict(
                     threading=False,
                     chrome_options={
                         # '--headless',
                         '--disable-gpu',
                         '--disable-dev-shm-usage',
                         '--no-sandbox',
                     }
                 ))

  env.reset()


if __name__ == '__main__':
  app.run(main)
