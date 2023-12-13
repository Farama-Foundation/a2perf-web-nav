# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy and value Networks for web navigation tasks."""

import gin
import sonnet as snt
import tensorflow as tf

PROFILE_KEY = "profile_key"
PROFILE_KEY_MASK = "profile_key_mask"
PROFILE_VALUE = "profile_value"
PROFILE_VALUE_MASK = "profile_value_mask"
DOM_ELEMENTS = "dom_elements"
DOM_ATTRIBUTE_MASK = "dom_attribute_mask"
DOM_FEATURES = "dom_features"
DOM_ELEMENTS_MASK = "dom_elements_mask"
DOM_PROFILE_JOINT_MASK = "dom_profile_joint_mask"


def masked_mean_aggregator(x, mask=None, axis=-2):
  """Function for aggregation over a given axis with masking.

  If mask is not given, this reduces to regular tf.reduce_mean.
  If mask is given, it is assumed that the last dimension is the actual vector
  dimension. Mask is always expanded in the last dimension.

  Args:
    x: An N-D tensor.
    mask: An (N-1)-D tensor.
    axis: Axis of the aggregation wrt x.

  Returns:
    Masked mean of the input tensor along the axis.
  """
  if mask is None:
    return tf.reduce_mean(x, axis=axis)
  return tf.math.divide_no_nan(
      tf.reduce_sum(x * tf.expand_dims(mask, axis=-1), axis=axis),
      tf.reduce_sum(mask, axis=axis + 1, keepdims=True))


def embed_and_aggregate(embedder, tokens, mask, axis):
  embedded_tokens = embedder(tokens)
  aggregated_tokens = masked_mean_aggregator(embedded_tokens, mask=mask,
                                             axis=axis)
  return aggregated_tokens


def encode_profile(profile_encoder, embedder, observation, training,
    profile_value_dropout_rate):
  profile_key_emb = embed_and_aggregate(embedder, observation[PROFILE_KEY],
                                        observation[PROFILE_KEY_MASK],
                                        axis=-2)
  profile_value_emb = embed_and_aggregate(embedder,
                                          observation[PROFILE_VALUE],
                                          observation[PROFILE_VALUE_MASK],
                                          axis=-2)

  if training and profile_value_dropout_rate < 1.0:
    profile_value_emb_shp = tf.shape(profile_value_emb)
    profile_value_emb = tf.nn.dropout(profile_value_emb,
                                      profile_value_dropout_rate,
                                      noise_shape=[profile_value_emb_shp[0],
                                                   profile_value_emb_shp[1], 1])
    profile_emb = tf.concat([profile_key_emb, profile_value_emb], axis=-1)
  elif profile_value_dropout_rate >= 1.0:
    profile_emb = profile_key_emb
  else:
    profile_emb = tf.concat([profile_key_emb, profile_value_emb], axis=-1)

  return profile_encoder(profile_emb)


def encode_dom(dom_element_encoder, dom_encoder, embedder, observation,
    embedding_dim,
    use_bidirectional_encoding, dom_encoder_bw=None, fw_bw_encoder=None):
  element_embeddings = embed_and_aggregate(embedder=embedder,
                                           tokens=observation[DOM_ELEMENTS],
                                           mask=observation[DOM_ATTRIBUTE_MASK],
                                           axis=-2)

  element_embeddings_reshaped = tf.reshape(element_embeddings, [
      -1, tf.shape(element_embeddings)[1],
      tf.shape(element_embeddings)[2] * embedding_dim
  ])

  element_encodings = dom_element_encoder(element_embeddings_reshaped)
  element_encodings = tf.concat(
      [element_encodings, observation[DOM_FEATURES]], axis=-1)
  dom_encoding, _ = dom_encoder(tf.transpose(element_encodings, [1, 0, 2]),
                                dom_encoder.initial_state(
                                    tf.shape(element_encodings)[0]))

  if use_bidirectional_encoding:
    element_encodings_bw, _ = dom_encoder_bw(
        tf.transpose(element_encodings, [1, 0, 2]),
        dom_encoder_bw.initial_state(tf.shape(element_encodings)[0]))
    dom_encoding = fw_bw_encoder(
        tf.concat([dom_encoding, element_encodings_bw], axis=-1))

  return tf.transpose(dom_encoding, [1, 0, 2])


@gin.configurable("DQNWebLSTM")
class DQNWebLSTM(snt.Module):
  """DQN with LSTM-based DOM encoder for web navigation."""

  def __init__(self,
      vocab_size,
      embedding_dim,
      latent_dim,
      number_of_dom_encoder_layers=1,
      number_of_profile_encoder_layers=1,
      flatten_output=True,
      embedding_initializer=None,
      profile_value_dropout=0.0,
      q_min=None,
      q_max=None,
      use_select_option_dim=False,
      name=None,
      return_state_value=False,
      predict_action_type=True,
      use_bidirectional_encoding=False):
    """DQN with LSTM-based DOM encoder.

    Profile and DOM are independently encoded into tensors where profile tensor
    represents field encodings while DOM tensor represents element encodings.
    DOM elements are encoded by running an LSTM network over element encodings.
    These two tensors are used score every element and field pairs. Scores
    correspond to Q values in DQN or logits in a policy network.

    Args:
      vocab_size: Size of the embeddings vocabulary. This should be large enough
        to accommodate novel vocabulary items while navigating the web page.
      embedding_dim: Dimensionality of the embedding tensor.
      latent_dim: Dimensionality of the latent space.
      number_of_dom_encoder_layers: Number of layers to use in feed forward
        layers in DOM encoder.
      number_of_profile_encoder_layers: Number of layers to use in feed forward
        layers in profile encoder.
      flatten_output: If true, flatten output Q value tensor into an array.
      embedding_initializer: Initializer for the embeddings.
      profile_value_dropout: Apply dropout on value level rather than dimension
        level.
      q_min: Minimum Q value for scaling.
      q_max: Maximum Q value for scaling.
      use_select_option_dim: If true, add an additional action type dimension
        for select action.
      name: Name of the layer.
      return_state_value: If true, estimate and return a state value prediction.
      predict_action_type: If true, action type is also predicted in addition to
        generating a joint distribution over elements and profile fields.
      use_bidirectional_encoding: If true, use BiLSTM encoder for DOM encoding.
    """
    super().__init__(name=name)
    self._embedding_dim = embedding_dim
    if number_of_dom_encoder_layers < 0:
      raise ValueError(
          ("Number of DOM encoder layers "
           "should be > 0 but got %d") % number_of_dom_encoder_layers)
    if number_of_profile_encoder_layers < 0:
      raise ValueError(("Number of profile encoder "
                        "layers should be > 0 but got"
                        " %d") % number_of_profile_encoder_layers)
    self._q_min = q_min
    self._q_max = q_max
    if self._q_min and self._q_max and self._q_min > self._q_max:
      raise ValueError(
          "Q value bounds are invalid: q_min({}) > q_max({}).".format(
              self._q_min, self._q_max))
    self._flatten_output = flatten_output
    self._predict_action_type = predict_action_type
    # Embedding matrix.
    self._embedder = snt.Embed(
        vocab_size, embedding_dim, initializer=embedding_initializer)
    # Independent DOM element encoder using multi-layer MLP. Each element is
    # encoded with this network before passing to BiLSTM encoder.
    self._dom_element_encoder = snt.Sequential([
        snt.nets.MLP(
            [latent_dim] * number_of_dom_encoder_layers,
            activation=tf.nn.relu,
            activate_final=True),
    ])
    self._use_bidirectional_encoding = use_bidirectional_encoding
    # Encode DOM tree globally via LSTM.
    self._dom_encoder = snt.UnrolledLSTM(latent_dim)
    if self._use_bidirectional_encoding:
      self._dom_encoder_bw = snt.UnrolledLSTM(latent_dim)
      self._fw_bw_encoder = snt.nets.MLP([latent_dim],
                                         activation=tf.identity,
                                         activate_final=False)
    # Profile encoder via multi-layer MLP.
    self._profile_encoder = snt.Sequential([
        snt.nets.MLP(
            [latent_dim] * number_of_profile_encoder_layers,
            activation=tf.nn.relu,
            activate_final=True),
    ])
    number_of_action_types = 2
    if use_select_option_dim:
      number_of_action_types += 1
    # Predict action type via multi-layer MLP.
    if self._predict_action_type:
      self._action_type_encoder = snt.Sequential([
          snt.nets.MLP([latent_dim, number_of_action_types],
                       activation=tf.nn.relu,
                       activate_final=False),
      ])
    self._profile_value_dropout_rate = profile_value_dropout
    self._return_state_value = return_state_value
    # Predict state value via an MLP.
    if return_state_value:
      self._value_network = snt.nets.MLP([latent_dim, 1],
                                         activation=tf.nn.relu,
                                         activate_final=False)

  def __call__(self, observation, training=True):
    """Compute Q values for web navigation tasks.

      Encodes flattened DOM elements using LSTM and outputs Q values using user
      profile and DOM element encodings.

    Args:
      observation: A nested observation (dictionary of observations) from web
        navigation environment.
      training: Is the model training. Required for applying dropout.

    Returns:
      Q values of the form (dom elements, action type, type sequence). If
      flatten_output is True, flatten this tuple into an array.
    """

    profile_enc = encode_profile(profile_encoder=self._profile_encoder,
                                 training=training,
                                 embedder=self._embedder,
                                 observation=observation,
                                 profile_value_dropout_rate=self._profile_value_dropout_rate)

    dom_encoding = encode_dom(dom_element_encoder=self._dom_element_encoder,
                              dom_encoder=self._dom_encoder,
                              observation=observation,
                              embedder=self._embedder,
                              embedding_dim=self._embedding_dim,
                              use_bidirectional_encoding=self._use_bidirectional_encoding,
                              dom_encoder_bw=self._dom_encoder_bw,
                              fw_bw_encoder=self._fw_bw_encoder)

    q_values_joint = tf.reduce_sum(
        tf.expand_dims(dom_encoding, axis=1) *
        tf.expand_dims(profile_enc, axis=2),
        axis=-1)  # (batch, fields, elements)
    q_values = tf.expand_dims(
        q_values_joint, axis=1)  # (batch, 1, fields, elements)

    # Compute Q values for selecting action type.
    if self._predict_action_type:
      q_values_action_type = tf.transpose(
          self._action_type_encoder(dom_encoding),
          [0, 2, 1])  # (batch, 2, elements)
      # Combine these two Q values
      q_values = q_values + tf.expand_dims(
          q_values_action_type, axis=2)  # (batch, 2, fields, elements)

    # Prune DOM and field scores jointly. This will also prune padded dom
    # elements.
    dom_profile_joint_mask = tf.expand_dims(
        observation[DOM_PROFILE_JOINT_MASK], axis=1)
    q_values = dom_profile_joint_mask * q_values - 999999. * (
        1 - dom_profile_joint_mask)

    # Scale scores based on input minimum and maximum values.
    if self._q_min and self._q_max:
      q_values = tf.sigmoid(q_values) * (self._q_max -
                                         self._q_min) + self._q_min

    # If the RL framework (tf-agents + acme) requires a flat vector for outputs
    # flatten scores here and unflatten in the web environment.
    if self._flatten_output:
      q_values_shape_prod = tf.math.reduce_prod(tf.shape(q_values)[1:4])
      q_values = tf.reshape(q_values, [-1, q_values_shape_prod])
    elif not self._predict_action_type:
      q_values = tf.squeeze(q_values, axis=1)

    ############################################################################
    # State value prediction.
    ############################################################################
    # Predict state value.
    if self._return_state_value:
      value = self._value_network(
          tf.reduce_sum(
              tf.expand_dims(
                  tf.reduce_sum(tf.nn.softmax(q_values_joint), axis=1), axis=-1)
              * dom_encoding,
              axis=1))
      return q_values, tf.squeeze(value, axis=-1)
    return q_values


class WebLSTMBase(tf.keras.layers.Layer):
  '''Base class for LSTM-based web navigation networks.'''

  def __init__(self,
      vocab_size,
      embedding_dim,
      latent_dim,
      embedder=None,
      dom_element_encoder=None,
      dom_encoder=None,
      dom_encoder_bw=None,
      fw_bw_encoder=None,
      profile_encoder=None,
      number_of_dom_encoder_layers=1,
      number_of_profile_encoder_layers=1,
      flatten_output=True,
      embedding_initializer=None,
      profile_value_dropout=0.0,
      use_select_option_dim=False,
      name=None,
      predict_action_type=True,
      use_bidirectional_encoding=False):
    """Actor network with LSTM-based DOM encoder.

    Profile and DOM are independently encoded into tensors where profile tensor
    represents field encodings while DOM tensor represents element encodings.
    DOM elements are encoded by running an LSTM network over element encodings.
    These two tensors are used score every element and field pairs. Scores
    correspond to Q values in DQN or logits in a policy network.

    Args:
      vocab_size: Size of the embeddings vocabulary. This should be large enough
        to accommodate novel vocabulary items while navigating the web page.
      embedding_dim: Dimensionality of the embedding tensor.
      latent_dim: Dimensionality of the latent space.
      number_of_dom_encoder_layers: Number of layers to use in feed forward
        layers in DOM encoder.
      number_of_profile_encoder_layers: Number of layers to use in feed forward
        layers in profile encoder.
      embedding_initializer: Initializer for the embeddings.
      profile_value_dropout: Apply dropout on value level rather than dimension
        level.
      use_select_option_dim: If true, add an additional action type dimension
        for select action.
      name: Name of the layer.
    """
    super(WebLSTMBase, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    if number_of_dom_encoder_layers < 0:
      raise ValueError(
          ("Number of DOM encoder layers "
           "should be > 0 but got %d") % number_of_dom_encoder_layers)
    if number_of_profile_encoder_layers < 0:
      raise ValueError(("Number of profile encoder "
                        "layers should be > 0 but got"
                        " %d") % number_of_profile_encoder_layers)
    self._flatten_output = flatten_output
    self._predict_action_type = predict_action_type
    # Embedding matrix.
    if embedder is None:
      self._embedder = snt.Embed(
          vocab_size, embedding_dim, initializer=embedding_initializer)
    else:
      self._embedder = embedder

    # Independent DOM element encoder using multi-layer MLP. Each element is
    # encoded with this network before passing to BiLSTM encoder.

    if dom_element_encoder is None:
      self._dom_element_encoder = snt.Sequential([
          snt.nets.MLP(
              [latent_dim] * number_of_dom_encoder_layers,
              activation=tf.nn.relu,
              activate_final=True),
      ], name='dom_element_encoder')
    else:
      self._dom_element_encoder = dom_element_encoder
    self._use_bidirectional_encoding = use_bidirectional_encoding
    # Encode DOM tree globally via LSTM.

    if dom_encoder is None:
      self._dom_encoder = snt.UnrolledLSTM(latent_dim, name='dom_encoder')
    else:
      self._dom_encoder = dom_encoder

    self._use_bidirectional_encoding = use_bidirectional_encoding
    if self._use_bidirectional_encoding:
      # Set or create bidirectional encoding components
      self._dom_encoder_bw = dom_encoder_bw or snt.UnrolledLSTM(latent_dim,
                                                                name='dom_encoder_bw')
      self._fw_bw_encoder = fw_bw_encoder or snt.nets.MLP([latent_dim],
                                                          activation=tf.identity,
                                                          activate_final=False,
                                                          name='fw_bw_encoder')
    else:
      self._dom_encoder_bw = None
      self._fw_bw_encoder = None

    # Set or create profile encoder
    if profile_encoder is None:
      self._profile_encoder = snt.Sequential([
          snt.nets.MLP(
              [latent_dim] * number_of_profile_encoder_layers,
              activation=tf.nn.relu,
              activate_final=True),
      ], name='profile_encoder')
    else:
      self._profile_encoder = profile_encoder
    number_of_action_types = 2
    if use_select_option_dim:
      number_of_action_types += 1
    # Predict action type via multi-layer MLP.
    if self._predict_action_type:
      self._action_type_encoder = snt.Sequential([
          snt.nets.MLP([latent_dim, number_of_action_types],
                       activation=tf.nn.relu,
                       activate_final=False),
      ], name='action_type_encoder')
    self._profile_value_dropout_rate = profile_value_dropout

  def __call__(self, observation, training=True):
    profile_enc = encode_profile(profile_encoder=self._profile_encoder,
                                 training=training,
                                 embedder=self._embedder,
                                 observation=observation,
                                 profile_value_dropout_rate=self._profile_value_dropout_rate)

    dom_encoding = encode_dom(embedder=self._embedder,
                              dom_element_encoder=self._dom_element_encoder,
                              dom_encoder=self._dom_encoder,
                              observation=observation,
                              embedding_dim=self._embedding_dim,
                              use_bidirectional_encoding=self._use_bidirectional_encoding,
                              dom_encoder_bw=self._dom_encoder_bw,
                              fw_bw_encoder=self._fw_bw_encoder)

    logits_joint = tf.reduce_sum(
        tf.expand_dims(dom_encoding, axis=1) *
        tf.expand_dims(profile_enc, axis=2),
        axis=-1)  # (batch, fields, elements)
    logits = tf.expand_dims(
        logits_joint, axis=1)  # (batch, 1, fields, elements)

    # Compute logits for selecting action type.
    if self._predict_action_type:
      logits_action_type = tf.transpose(
          self._action_type_encoder(dom_encoding),
          [0, 2, 1])  # (batch, 2, elements)
      # Combine these two
      logits = logits + tf.expand_dims(
          logits_action_type, axis=2)  # (batch, 2, fields, elements)

    # Prune DOM and field scores jointly. This will also prune padded dom
    # elements.
    dom_profile_joint_mask = tf.expand_dims(
        observation[DOM_PROFILE_JOINT_MASK], axis=1)
    logits = dom_profile_joint_mask * logits - 999999. * (
        1 - dom_profile_joint_mask)

    # If the RL framework (tf-agents + acme) requires a flat vector for outputs
    # flatten scores herde and unflatten in the web environment.
    if self._flatten_output:
      logits_shape_prod = tf.math.reduce_prod(tf.shape(logits)[1:4])
      logits = tf.reshape(logits, [-1, logits_shape_prod])
    elif not self._predict_action_type:
      logits = tf.squeeze(logits, axis=1)

    return logits


class WebLSTMActor(WebLSTMBase):

  def __init__(self,
      vocab_size,
      embedding_dim,
      latent_dim,
      embedder=None,
      dom_element_encoder=None,
      dom_encoder=None,
      profile_encoder=None,
      fw_bw_encoder=None,
      dom_encoder_bw=None,
      number_of_dom_encoder_layers=1,
      number_of_profile_encoder_layers=1,
      flatten_output=True,
      embedding_initializer=None,
      profile_value_dropout=0.0,
      use_select_option_dim=False,
      name=None,
      predict_action_type=True,
      use_bidirectional_encoding=False):
    super(WebLSTMActor, self).__init__(vocab_size=vocab_size,
                                       embedding_dim=embedding_dim,
                                       latent_dim=latent_dim, embedder=embedder,
                                       dom_element_encoder=dom_element_encoder,
                                       dom_encoder=dom_encoder,
                                       dom_encoder_bw=dom_encoder_bw,
                                       fw_bw_encoder=fw_bw_encoder,
                                       profile_encoder=profile_encoder,
                                       number_of_dom_encoder_layers=number_of_dom_encoder_layers,
                                       number_of_profile_encoder_layers=number_of_profile_encoder_layers,
                                       flatten_output=flatten_output,
                                       embedding_initializer=embedding_initializer,
                                       profile_value_dropout=profile_value_dropout,
                                       use_select_option_dim=use_select_option_dim,
                                       name=name,
                                       predict_action_type=predict_action_type,
                                       use_bidirectional_encoding=use_bidirectional_encoding)

  def __call__(self, observation, training=True):
    """Compute probabilities for web navigation tasks.

        Encodes flattened DOM elements using LSTM and outputs probabilities using
        user profile and DOM element encodings.
    """
    logits = super().__call__(observation, training)
    return logits


class WebLSTMCritic(WebLSTMBase):

  def __init__(self,
      vocab_size,
      embedding_dim,
      latent_dim,
      embedder=None,
      dom_element_encoder=None,
      fw_bw_encoder=None,
      dom_encoder_bw=None,
      dom_encoder=None,
      profile_encoder=None,
      number_of_dom_encoder_layers=1,
      number_of_profile_encoder_layers=1,
      flatten_output=True,
      embedding_initializer=None,
      profile_value_dropout=0.0,
      use_select_option_dim=False,
      name=None,
      predict_action_type=True,
      use_bidirectional_encoding=False):
    super(WebLSTMCritic, self).__init__(vocab_size=vocab_size,
                                        embedding_dim=embedding_dim,
                                        latent_dim=latent_dim,
                                        embedder=embedder,
                                        dom_element_encoder=dom_element_encoder,
                                        dom_encoder=dom_encoder,
                                        dom_encoder_bw=dom_encoder_bw,
                                        fw_bw_encoder=fw_bw_encoder,
                                        profile_encoder=profile_encoder,
                                        number_of_dom_encoder_layers=number_of_dom_encoder_layers,
                                        number_of_profile_encoder_layers=number_of_profile_encoder_layers,
                                        flatten_output=flatten_output,
                                        embedding_initializer=embedding_initializer,
                                        profile_value_dropout=profile_value_dropout,
                                        use_select_option_dim=use_select_option_dim,
                                        name=name,
                                        predict_action_type=predict_action_type,
                                        use_bidirectional_encoding=use_bidirectional_encoding)

    self._value_network = snt.nets.MLP([latent_dim, 1],
                                       activation=tf.nn.relu,
                                       activate_final=False)

  def __call__(self, observation, training=True):
    """Computes state value for web navigation tasks.
    """
    logits = super().__call__(observation, training)
    value = self._value_network(logits)
    return tf.squeeze(value, axis=-1)


@gin.configurable("DQNWebFF")
class DQNWebFF(snt.Module):
  """Feed forward DQN for web navigation."""

  def __init__(self,
      vocab_size,
      embedding_dim,
      latent_dim,
      number_of_dom_encoder_layers=1,
      number_of_profile_encoder_layers=1,
      embedding_initializer=None,
      profile_value_dropout=0.0,
      use_select_option_dim=False,
      name=None,
      return_state_value=False):
    """DQN with feed forward DOM encoder.

    Profile and DOM are independently encoded into tensors where profile tensor
    represents field encodings while DOM tensor represents element encodings.
    These two tensors are used score every element and field pairs. Scores
    correspond to Q values in DQN or logits in a policy network.

    Args:
      vocab_size: Size of the embeddings vocabulary. This should be large enough
        to accommodate novel vocabulary items while navigating the web page.
      embedding_dim: Dimensionality of the embedding tensor.
      latent_dim: Dimensionality of the latent space.
      number_of_dom_encoder_layers: Number of layers to use in feed forward
        layers in DOM encoder.
      number_of_profile_encoder_layers: Number of layers to use in feed forward
        layers in profile encoder.
      embedding_initializer: Initializer for the embeddings.
      profile_value_dropout: Apply dropout on value level rather than dimension
        level.
      use_select_option_dim: If true, use another dimension on Q values for
        select_option action.
      name: Name of the layer.
      return_state_value: If true, return a value output as well for IMPALA
        training.
    """
    super().__init__(name=name)
    self._embedding_dim = embedding_dim
    if number_of_dom_encoder_layers < 0:
      raise ValueError(
          ("Number of DOM encoder layers "
           "should be > 0 but got %d") % number_of_dom_encoder_layers)
    if number_of_profile_encoder_layers < 0:
      raise ValueError(("Number of profile encoder "
                        "layers should be > 0 but got"
                        " %d") % number_of_profile_encoder_layers)
    # Embedding matrix.
    self._embedder = snt.Embed(
        vocab_size, embedding_dim, initializer=embedding_initializer)
    # Independent DOM element encoder using multi-layer MLP.
    self._dom_element_encoder = snt.Sequential([
        snt.nets.MLP(
            [latent_dim] * number_of_dom_encoder_layers,
            activation=tf.nn.relu,
            activate_final=True),
    ])
    # Final DOM encoder that blends element encodings with float array features.
    self._dom_encoder = snt.nets.MLP([latent_dim],
                                     activation=tf.nn.relu,
                                     activate_final=True)
    # Profile encoder via multi-layer MLP.
    self._profile_encoder = snt.Sequential([
        snt.nets.MLP(
            [latent_dim] * number_of_profile_encoder_layers,
            activation=tf.nn.relu,
            activate_final=True),
    ])
    number_of_action_types = 2
    if use_select_option_dim:
      number_of_action_types += 1
    # Predict action type via multi-layer MLP.
    self._action_type_encoder = snt.Sequential([
        snt.nets.MLP([latent_dim, number_of_action_types],
                     activation=tf.nn.relu,
                     activate_final=False),
    ])
    self._profile_value_dropout_rate = profile_value_dropout
    self._return_state_value = return_state_value
    # Predict state value via an MLP.
    if return_state_value:
      self._value_network = snt.nets.MLP([latent_dim, 1],
                                         activation=tf.nn.relu,
                                         activate_final=False)

  def __call__(self, observation, training=True):
    """Compute Q values for web navigation tasks.

      Encodes flattened DOM elements using LSTM and outputs Q values using user
      profile and DOM element encodings.

    Args:
      observation: A nested observation (dictionary of observations) from web
        navigation environment.
      training: Is the model training. Required for applying dropout.

    Returns:
      Q values of the form (dom elements, action type, type sequence). If
      flatten_output is True, flatten this tuple into an array.
    """
    ############################################################################
    # Profile encoder.
    ############################################################################
    # Embed profile keys and values.
    profile_key_emb = masked_mean_aggregator(
        self._embedder(observation[PROFILE_KEY]),
        mask=observation[PROFILE_KEY_MASK],
        axis=-2)  # (batch, fields, tokens, dim) --> (batch, fields, dim)
    profile_value_emb = masked_mean_aggregator(
        self._embedder(observation[PROFILE_VALUE]),
        mask=observation[PROFILE_VALUE_MASK],
        axis=-2)  # (batch, fields, tokens, dim) --> (batch, fields, dim)
    if training:
      profile_value_emb_shp = tf.shape(profile_value_emb)
      profile_value_emb = tf.nn.dropout(
          profile_value_emb,
          self._profile_value_dropout_rate,
          noise_shape=[profile_value_emb_shp[0], profile_value_emb_shp[1], 1])
    # Concat profile keys and values and encode via a feed forward network.
    profile_emb = tf.concat([profile_key_emb, profile_value_emb],
                            axis=-1)  # (batch, fields, 2*dim)
    profile_enc = self._profile_encoder(profile_emb)  # (batch, fields, dim)

    ############################################################################
    # DOM encoder.
    ############################################################################
    # Embed dom elements via lookup and aggregation over tokens.
    # (batch, elements, attributes, tokens, dim) -->
    #     (batch, elements, attributes, dim)
    element_embeddings = masked_mean_aggregator(
        self._embedder(observation[DOM_ELEMENTS]),
        mask=observation[DOM_ATTRIBUTE_MASK],
        axis=-2)  # (batch, elements, attributes, dim)
    element_embeddings_reshaped = tf.reshape(element_embeddings, [
        -1,
        tf.shape(element_embeddings)[1],
        tf.shape(element_embeddings)[2] * self._embedding_dim
    ])  # (batch, elements, attributes*dim)

    # Encode dom elements using a feed forward network.
    element_encodings = self._dom_element_encoder(
        element_embeddings_reshaped)  # (batch, elements, dim)
    element_encodings = tf.concat(
        [element_encodings, observation[DOM_FEATURES]],
        axis=-1)  # (batch, elements, dim+features)
    element_encodings = self._dom_encoder(
        element_encodings)  # (elements, batch, dim)

    ############################################################################
    # Pairwise scoring.
    ############################################################################
    # Compute pairwise scores (or joint Q values or joint distribution) over
    # every DOM element and profile field pairs.
    q_values_joint = tf.reduce_sum(
        tf.expand_dims(element_encodings, axis=1) *
        tf.expand_dims(profile_enc, axis=2),
        axis=-1)  # (batch, fields, elements)
    # Prune DOM and field scores jointly. This will also prune padded dom
    # elements.
    q_values_joint = observation[
                       DOM_PROFILE_JOINT_MASK] * q_values_joint - 999999. * (
                         1. - observation[DOM_PROFILE_JOINT_MASK])
    q_values = tf.reduce_sum(q_values_joint, axis=1)
    ############################################################################
    # State value prediction.
    ############################################################################
    # Predict state value.
    if self._return_state_value:
      value = self._value_network(
          tf.reduce_sum(
              tf.expand_dims(
                  tf.reduce_sum(tf.nn.softmax(q_values_joint), axis=1), axis=-1)
              * element_encodings,
              axis=1))
      return q_values, tf.squeeze(value, axis=-1)
    return q_values
