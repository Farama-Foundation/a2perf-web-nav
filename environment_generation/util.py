from CoDE import web_environment
from CoDE import web_primitives
from CoDE import vocabulary_node

import collections


def measure_gMiniWob_difficulty(design):
    """Measures the difficulty of a webpage design with respect to gMiniWob."""
    num_pages = design['number_of_pages']
    action_page = design['action_page']

    # Probability of a random agent interacting with the correct element

    action_counts = collections.Counter(action_page)
    prob = 1.0 / 1

    # Difficulty of the active primitives
    primitive_difficulty = 1

    # Ratio of the number of active/passive primitives
    ratio_difficulty = 1
    return ratio_difficulty + prob + primitive_difficulty


def measure_html_difficulty(html):
    """Measures the difficulty of a webpage design."""
    return 0


def measure_difficulty(design):
    """Measures the difficulty of a webpage design."""

    # Create a webpage from the design.
    env = web_environment.GMiniWoBWebEnvironment(
            base_url="file:///path/to/compositional_rl/gwob/",
            global_vocabulary=vocabulary_node.LockedVocabulary())

    # Diffulty measure will be a weighted average of gMiniWob and HTML difficulty.
    gMiniWob_difficulty = measure_gMiniWob_difficulty(design)
    html_difficulty = measure_html_difficulty(design)
    return 0.5 * gMiniWob_difficulty + 0.5 * html_difficulty


if __name__ == '__main__':
    d = {'number_of_pages': 4,
         'action_page': [0, 0, 0, 0],
         'action': [13, 13, 13, 13]}
    print(measure_difficulty(d))
