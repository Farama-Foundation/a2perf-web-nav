import json
import socket
import time
import traceback
import urllib.parse
from queue import Queue
from threading import Thread

import numpy as np
from absl import logging
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.driver_cache import DriverCacheManager

from a2perf.domains.web_navigation.gwob.miniwob_plusplus.python.miniwob.fields import \
  Fields
from a2perf.domains.web_navigation.gwob.miniwob_plusplus.python.miniwob.fields import \
  get_field_extractor
from a2perf.domains.web_navigation.gwob.miniwob_plusplus.python.miniwob.reward import \
  get_original_reward
from a2perf.domains.web_navigation.gwob.miniwob_plusplus.python.miniwob.screenshot import \
  get_screenshot
from a2perf.domains.web_navigation.gwob.miniwob_plusplus.python.miniwob.state import \
  MiniWoBState


def find_free_port():
  """Finds an available port on the local machine."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]


class MiniWoBInstance(Thread):
  """Interface between Python and Chrome driver via Selenium.
  Manages a single instance.
  """

  DEFAULT_BASE_URL = 'http://localhost:8000/'

  # Added some space for title bar
  WINDOW_WIDTH = 500
  WINDOW_HEIGHT = 240
  TASK_WIDTH = 160
  TASK_HEIGHT = 210

  FLIGHT_WINDOW_WIDTH = 600
  FLIGHT_WINDOW_HEIGHT = 700
  FLIGHT_TASK_WIDTH = 375
  FLIGHT_TASK_HEIGHT = 667

  WINDOW_POSITION_X = 9000
  WINDOW_POSITION_Y_OFFSET = 30

  def __init__(self, index, subdomain, seed,
      base_url=None, cache_state=False, threading=False, chrome_options=None,
      reward_processor=None, wait_ms=0., block_on_reset=True,
      refresh_freq=0, initial_mode='train'):
    """Starts a new Selenium WebDriver session.

    Args:
        index (int): Instance index
        subdomain (str): MiniWoB task name (e.g., "click-test")
        seed (object): Random seed
        base_url (str): Base URL (default to localhost at port 8000)
        cache_state (bool): Whether to cache and return the initial
            state; only make sense if the task interface never changes
        threading (bool): Whether to run this instance as a Thread
        chrome_options (list[str]): Additional Chrome options
        reward_processor (callable; optional): A function that takes
            the metadata and return a reward (see miniwob.reward)
        wait_ms (float): Pause the instance after each action for this
            amount of time (in milliseconds).
        block_on_reset (bool): On reset, block until the page loads.
        refresh_freq (int): Every this number of episodes,
            refresh the page at the beginning of the next episode.
            Takes time but cleans up any lingering states and memory leaks.
            *** Must specify `seeds` at each reset call.
        initial_mode (str): Initial data mode (e.g., "train", "test")
    """
    super(MiniWoBInstance, self).__init__()
    # Overrides Thread.daemon: Kill this thread when the parent is killed
    self.daemon = True
    self.died = False
    self.index = index
    self.init_seed = repr(seed)
    self.chrome_options = chrome_options
    print('base_url: ', base_url)
    base_url = base_url or self.DEFAULT_BASE_URL
    print('base_url: ', base_url)
    if subdomain.startswith('flight.'):
      assert not base_url.startswith('file://'), \
        ('For {} domain, MINIWOB_BASE_URL cannot be file://. '
         ' See "Run a simple server" in README').format(subdomain)
      self.url = urllib.parse.urljoin(base_url,
                                      subdomain.replace('.',
                                                        '/') + '/wrapper.html')
      self.window_width = self.FLIGHT_WINDOW_WIDTH
      self.window_height = self.FLIGHT_WINDOW_HEIGHT
      self.task_width = self.FLIGHT_TASK_WIDTH
      self.task_height = self.FLIGHT_TASK_HEIGHT

    elif subdomain.startswith('gminiwob.') or subdomain.startswith('gwob.'):
      self.url = urllib.parse.urljoin(base_url, '{}/{}.html'.format(
          subdomain[0:subdomain.index('.')],
          subdomain[subdomain.index('.') + 1:]))
      print('url: ', self.url)
      self.window_width = self.FLIGHT_WINDOW_WIDTH
      self.window_height = self.FLIGHT_WINDOW_HEIGHT
      self.task_width = self.FLIGHT_TASK_WIDTH
      self.task_height = self.FLIGHT_TASK_HEIGHT
    else:
      self.url = urllib.parse.urljoin(base_url,
                                      'miniwob/{}.html'.format(subdomain))
      self.window_width = self.WINDOW_WIDTH
      self.window_height = self.WINDOW_HEIGHT
      self.task_width = self.TASK_WIDTH
      self.task_height = self.TASK_HEIGHT
    self.field_extractor = get_field_extractor(subdomain)
    self.cache_state = cache_state
    self.threading = threading
    self.reward_processor = reward_processor
    self.wait_ms = wait_ms
    self.block_on_reset = block_on_reset
    self.refresh_freq = refresh_freq
    self.num_episodes = 0
    self.mode = initial_mode
    self.record_screenshots = False
    if reward_processor is None:
      # Use the original reward
      self.reward_processor = get_original_reward
    self.start_time = float('inf')
    self.task_queue = Queue()
    if not threading:
      # Hack: override the start method of Thread
      self.start = self.create_driver

  def run(self):
    """Overrides `Thread.run`"""
    try:
      self.create_driver()
      # Wait for command
      while True:
        func, args = self.task_queue.get()
        try:
          func(*args)
        except Exception as e:
          logging.error('Error in instance %d', self.index)
          traceback.print_exc()
          self.died = True
        self.task_queue.task_done()
        if func == self.close:
          break
    finally:
      self.close()
      logging.info('Closed instance %d', self.index)

  def call(self, func, *args):
    if self.threading:
      self.task_queue.put((func, args))
    else:
      func(*args)

  def wait(self):
    if self.threading:
      self.task_queue.join()

  ################################
  # Possible Functions
  def create_driver(self):
    """
    Create a Chrome WebDriver instance for the class.
    This method handles driver caching, initializes options based on class properties,
    and manages the WebDriver instance for further use.
    """
    assert not hasattr(self,
                       'driver'), f'Instance {self.index} already has a driver'
    options = self._configure_driver_options()

    chromedriver_binary_path = self._get_driver_path()
    service = Service(executable_path=chromedriver_binary_path)
    self.driver = webdriver.Chrome(service=service, options=options)
    self._initialize_driver_session()

  def _configure_driver_options(self):
    options = webdriver.ChromeOptions()

    headless = self.chrome_options and '--headless' in self.chrome_options
    if not headless:
      self._set_rendering_options(options)

    if self.chrome_options:
      for opt in self.chrome_options:
        options.add_argument(opt)

    return options

  def _set_rendering_options(self, options):
    options.add_argument('--use-gl=swiftshader')
    options.add_argument(f'app={self.url}')
    window_position_y = self.WINDOW_POSITION_Y_OFFSET + self.index * (
        self.window_height + self.WINDOW_POSITION_Y_OFFSET)
    options.add_argument(
        f'window-size={self.window_width},{self.window_height}')
    options.add_argument(
        f'window-position={self.WINDOW_POSITION_X},{window_position_y}')

  def _get_driver_path(self):
    driver_cache_manager = DriverCacheManager()
    driver_cache_manager_metadata = driver_cache_manager.load_metadata_content()

    if driver_cache_manager_metadata:
      latest_driver = max(driver_cache_manager_metadata.items(),
                          key=lambda x: x[1]['timestamp'])
      chromedriver_binary_path = latest_driver[1]['binary_path']
      logging.info(f'Using cached driver at {chromedriver_binary_path}')
    else:
      logging.info('No cached driver found, downloading latest version')
      chromedriver_binary_path = ChromeDriverManager().install()

    return chromedriver_binary_path

  def _initialize_driver_session(self):
    self.driver.implicitly_wait(10)
    headless = '--headless' in self.chrome_options if self.chrome_options else False
    if headless:
      self.driver.get(self.url)

    try:
      WebDriverWait(self.driver, 5).until(
          EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID)))
    except TimeoutException as e:
      logging.error('Page did not load properly. Wrong MINIWOB_BASE_URL?')
      raise e

    self.driver.execute_script(f'Math.seedrandom({self.init_seed});')

  def close(self):
    """Tear down the WebDriver."""
    # Note: close() will close the current window
    # quit() closes everything, so it is probably cleaner
    try:
      self.driver.quit()
    except Exception as e:
      logging.error('Error closing the driver of instance %d', self.index)
      traceback.print_exc()
    self.died = True

  def reset(self, states, seed):
    """Forces stop and start this instance.
    Also sets states[i] to be the initial state
    (where i = self.index).

    Args:
        states (list)
        seed (object): Seed to set for the next episode
    """
    if self.refresh_freq:
      assert seed is not None, 'reset() must specify seed if refresh_freq is specified'
    i = self.index
    self.force_stop()
    self.begin_task(seed=seed)
    states[i] = self.get_state()
    if self.cache_state:
      self.initial_state = states[i]

  def step(self, action, states, rewards, dones, info_n):
    """Applies an action on this instance.
    Also sets states[i], rewards[i], dones[i], and info['n'][i]
    (where i = self.index).

    Args:
        action (MiniWoBAction)
        states (list)
        rewards (list)
        dones (list)
        info_n (list)
    """
    i = self.index
    self.perform(action)
    metadata = self.get_metadata()
    rewards[i] = self.reward_processor(metadata)
    dones[i] = metadata['done']
    if not metadata['done']:
      if not self.cache_state:
        states[i] = self.get_state()
      else:
        states[i] = self.initial_state
    metadata['elapsed'] = max(0., time.time() - self.start_time)
    info_n[i] = metadata

  ################################
  # Primitive actions

  SYNC_SCREEN_ID = 'sync-task-cover'
  RESET_BLOCK_SLEEP_TIME = 0.05  # 50ms
  RESET_BLOCK_MAX_ATTEMPT = 20  # up to 1s

  def force_stop(self):
    """Force stop the task and go back to the sync screen."""
    self.driver.execute_script('return core.endEpisode(0);')

  def begin_task(self, seed=None):
    """Start the task. Only available when done is True.
    The sync screen will disappear and the countdown timer will start.

    Args:
        seed: New seed to set for the next episode
    """
    self.num_episodes += 1
    if self.refresh_freq and self.num_episodes % self.refresh_freq == 0:
      self.driver.get(self.url)
    if seed is not None:
      self.set_seed(seed)
    self.set_mode(self.mode)
    # Wait for the sync screen, then click on it
    # element = WebDriverWait(self.driver, 5).until(
    #                EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID)))
    # self.driver.find_element_by_id(self.SYNC_SCREEN_ID).click()
    self.driver.execute_script('core.startEpisodeReal();')
    if self.block_on_reset:
      for _ in range(self.RESET_BLOCK_MAX_ATTEMPT):
        if self.driver.execute_script('return WOB_TASK_READY;'):
          break
        time.sleep(self.RESET_BLOCK_SLEEP_TIME)
      else:
        raise RuntimeError(
            'Instance {} does not load properly'.format(self.index))
    elif self.wait_ms:
      time.sleep(self.wait_ms / 1000.)
    self.start_time = time.time()

  def perform(self, action):
    """Perform an action.

    Args:
        action: One of the following
        - None: Do nothing
        - a callable f(driver) that takes a Selenium driver as an argument;
            issue a warning if the instance is done
    """
    if action is not None:
      if self.get_metadata()['done']:
        logging.warn('Cannot call %s on instance %d, which is already done',
                     action, self.index)
      else:
        action(self.driver)
    if self.wait_ms:
      time.sleep(self.wait_ms / 1000.)

  def get_state(self):
    """Get the current state.

    Returns:
        MiniWoBState
    """
    # Get the utterance
    response = self.driver.execute_script('return core.getUtterance();')
    if isinstance(response, dict):
      utterance = response['utterance']
      fields = Fields(response['fields'])
    else:
      utterance = response
      fields = self.field_extractor(utterance)
    # Get the DOM
    dom_info = self.driver.execute_script('return core.getDOMInfo();')
    state = MiniWoBState(utterance, fields, dom_info)
    # Get screenshot if requested
    if self.record_screenshots:
      img = get_screenshot(self.driver, self.task_width, self.task_height)
      state.set_screenshot(img)
    return state

  def get_metadata(self):
    """Get other metadata.

    Returns:
        dict with the following keys:
        - done (bool)
        - env_reward (float; only well-defined when done is True):
            Environment-defined reward, possibly scaled by time
        - raw_reward (float; only well-defined when done is True):
            Environment-defined reward, NOT scaled by time
        - reason (any): reason for giving the reward (for debugging);
            will likely be None if done is False
    """
    return self.driver.execute_script(
        'return {'
        '"done": WOB_DONE_GLOBAL,'
        '"env_reward": WOB_REWARD_GLOBAL,'
        '"raw_reward": WOB_RAW_REWARD_GLOBAL,'
        '"reason": WOB_REWARD_REASON,'
        '};')

  def visualize_attention(self, attention):
    """Sends the attention weights to be visualized.

    Args:
        attentions: one of the following:
            - None: Do not do anything
            - np.array or 2d list of shape (num_grid_rows, num_grid_cols)
            - np.array or 2d list of shape (0, 0): Clear the visualization
    """
    if attention is None:
      return
    # Encode as JSON
    if isinstance(attention, np.ndarray):
      attention = attention.tolist()
    encoded = json.dumps(attention)
    # Send to the driver
    self.driver.execute_script('core.visualizeAttention({});'.format(encoded))

  def set_seed(self, seed):
    """Set the seed to a new value.

    Args:
        seed (object)
    """
    self.driver.execute_script('Math.seedrandom({});'.format(repr(seed)))

  def set_mode(self, mode):
    """Set the task generation mode (e.g., "train" or "test") to a new value.

    Args:
        mode (str)
    """
    self.driver.execute_script('core.setDataMode("{}");'.format(mode))
