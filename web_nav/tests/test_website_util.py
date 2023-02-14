import unittest

from web_nav.environment_generation import Primitive, Page, Website


class TestPage(unittest.TestCase):
    def test_init(self):
        page = Page(0)
        self.assertEqual(page.page_id, 0)
        self.assertIsNone(page.next_page)
        self.assertEqual(page.num_passive_primitives, 0)
        self.assertEqual(page.num_active_primitives, 0)
        self.assertEqual(page.num_interactable_elements, 0)
        self.assertEqual(page.num_possible_correct_interactions, 0)
        self.assertEqual(page.primitives, [])
        self.assertIsNone(page.difficulty)

    def test_add_primitive(self):
        page = Page(0)
        primitive = Primitive(name='header', primitive_id=0, num_interactable_elements=1, is_active=True)
        page.add_primitive(primitive)
        self.assertEqual(page.primitives, [primitive])
        self.assertEqual(page.num_passive_primitives, 0)
        self.assertEqual(page.num_active_primitives, 1)
        self.assertEqual(page.num_interactable_elements, 1)
        self.assertEqual(page.num_possible_correct_interactions, 1)
        self.assertIsNotNone(page.difficulty)

    def test_str(self):
        page = Page(0)
        self.assertEqual(str(page),
                         f'Page 0. Difficulty: None. Active primitives: 0. Passive primitives: 0 Next page: None')

    def test_repr(self):
        page = Page(0)
        self.assertEqual(repr(page),
                         f'Page 0. Difficulty: None. Active primitives: 0. Passive primitives: 0 Next page: None')

    def test_update_difficulty(self):
        page = Page(0)
        primitive = Primitive(name='header', primitive_id=0, num_interactable_elements=1, is_active=True)
        page.add_primitive(primitive)
        self.assertEqual(page.update_difficulty(), 0.0)

    def test_set_next_page(self):
        page1 = Page(0)
        page2 = Page(1)
        page1.set_next_page(page2)
        self.assertEqual(page1.next_page, page2)


class TestWebsite(unittest.TestCase):
    def test_init(self):
        page = Page(0)
        website = Website(page)
        self.assertEqual(website.first_page, page)
        self.assertEqual(website._last_page, page)
        self.assertIsNone(website.difficulty)
        self.assertEqual(website._num_pages, 1)

    def test_add_page(self):
        page1 = Page(0)
        page2 = Page(1)
        page1.difficulty = 1
        page2.difficulty = 1
        website = Website(page1)
        website.add_page(page2)
        self.assertEqual(website.difficulty, 2)
        self.assertEqual(website.first_page, page1)
        self.assertEqual(website._last_page, page2)
        self.assertEqual(website._num_pages, 2)

    def test_to_design(self):
        page1 = Page(0)
        primitive1 = Primitive(name='header', primitive_id=24, num_interactable_elements=3, is_active=True)
        page1.add_primitive(primitive1)
        page2 = Page(1)
        primitive2 = Primitive(name='header', primitive_id=33, num_interactable_elements=10, is_active=False)
        page2.add_primitive(primitive2)
        website = Website(page1)
        website.add_page(page2)
        design = website.to_design()
        self.assertEqual(design['number_of_pages'], 2)
        self.assertEqual(design['action'], [24, 33])
        self.assertEqual(design['action_page'], [0, 1])


class TestPrimitive(unittest.TestCase):
    def setUp(self):
        self.passive_primitive = Primitive("carousel", 1, 7)
        self.active_primitive = Primitive("username", 2, 1, True)

    def test_init(self):
        self.assertEqual(self.passive_primitive.name, "carousel")
        self.assertEqual(self.passive_primitive.primitive_id, 1)
        self.assertFalse(self.passive_primitive.is_active)
        self.assertEqual(self.passive_primitive.num_interactable_elements, 7)

        self.assertEqual(self.active_primitive.name, "username")
        self.assertEqual(self.active_primitive.primitive_id, 2)
        self.assertTrue(self.active_primitive.is_active)
        self.assertEqual(self.active_primitive.num_interactable_elements, 1)

    def test_str(self):
        self.assertEqual(str(self.passive_primitive), "carousel:1. Active: False")
        self.assertEqual(str(self.active_primitive), "username:2. Active: True")

    def test_repr(self):
        self.assertEqual(repr(self.passive_primitive), "carousel:1. Active: False")
        self.assertEqual(repr(self.active_primitive), "username:2. Active: True")


if __name__ == '__main__':
    unittest.main()
