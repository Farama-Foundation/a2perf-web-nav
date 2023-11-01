import gin
from absl import app

from rl_perf.domains.web_nav.compositional_rl.gwob.CoDE import vocabulary_node
from rl_perf.domains.web_nav.compositional_rl.gwob.CoDE import web_environment
from rl_perf.domains.web_nav.compositional_rl.gwob.CoDE import web_primitives


def main(_):
    gin.parse_config_files_and_bindings(["/web_nav/gwob/configs/envdesign.gin"], None)

    # Create an empty environment.
    env = web_environment.GMiniWoBWebEnvironment(
            base_url="file:///web_nav/gwob/",
            global_vocabulary=vocabulary_node.LockedVocabulary())

    ordered_concepts = web_primitives.CONCEPTS

    design = {'number_of_pages': len(ordered_concepts),
              'action': [web_primitives.CONCEPTS.index(x) for x in ordered_concepts],
              'action_page': range(len(ordered_concepts)),
              }

    # Design the actual environment.
    env.design_environment(
            design, auto_num_pages=True)


if __name__ == '__main__':
    app.run(main)
