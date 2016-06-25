import theano.tensor as T
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed, GRU, merge, Flatten, Embedding, SimpleRNN, Print, Recurrent, \
    Lambda, Reshape
import theano
import numpy as np


def feedback(model, output_length, feedback_function=lambda x: x,
             unroll=False, num_outputs=None):
    def feedback_loop(input):
        input = K.expand_dims(input, dim=1)
        if unroll:
            outputs = []
            for i in range(output_length):
                output = model(input)

                # needs to be a list for zip later
                if type(output) not in (list, tuple):
                    output = [output]

                outputs.append(output)
                input = feedback_function(output)
                for layer in model.layers:
                    if isinstance(layer, Recurrent):
                        layer.initial_state = layer.final_states
            return [K.concatenate(y, axis=1) for y in zip(*outputs)]
        else:
            assert num_outputs is not None

            def _step(tensor):
                output = [K.squeeze(output, axis=1) for output in model(tensor)]
                for layer in model.layers:
                    if issubclass(type(layer), Recurrent):
                        # TODO this is not being entered!
                        layer.initial_state = layer.final_states
                feedback_result = feedback_function(output)
                if type(output) not in (list, tuple):
                    output = [output]
                return output + [feedback_result]

            outputs, _ = theano.scan(_step,
                                     outputs_info=[None] * num_outputs + [input],
                                     n_steps=output_length)
            return outputs[:-1]

    (batch_size, _, output_dim) = model.input._keras_shape
    input = Input(batch_shape=(batch_size, output_dim))
    # outputs = []
    # for i in range(num_outputs):
    #     def get_ith_output(tensor):
    #         output = feedback_loop(tensor)
    #         return output[i]
    #     outputs.append(Lambda(get_ith_output, name='get_{}th_output'.format(i))(input))
    feedback_layer = Lambda(feedback_loop,
                            output_shape=([output_length, output_dim],
                                          [output_length, output_dim]))
    layer = feedback_layer(input)
    return Model(input, layer, name='feedback_model')
    # return Model(input, feedback_loop(input))


batch_size = 2
hidden_dim = 4
decoder_dim = 4
train_y_dim = 6
timesteps_i = 3
timesteps_o = 1
voc_size = 9
depth = 3

X_batch = np.zeros(batch_size * timesteps_i, dtype=int).reshape(
    batch_size, timesteps_i)
Y_batch = np.zeros(batch_size * timesteps_o, dtype=int).reshape(
    batch_size, timesteps_o, )

x = Input(shape=(timesteps_i,), dtype='int32', name='x')

embed = Embedding(voc_size, hidden_dim,
                  input_length=timesteps_i)
embed_lookup = embed(x)
gru = GRU(hidden_dim,
          input_shape=(timesteps_i, hidden_dim),
          unroll=True,
          return_sequences=True,
          consume_less='gpu')
encoder = Sequential([embed])
for _ in range(depth):
    encoder.add(gru)
gru_output = encoder(x)
attention = TimeDistributed(Dense(1))(gru_output)

# hidden_state = Lambda(lambda l: K.sum(l, axis=1))(gru(x))
hidden_state = Flatten()(merge([gru_output, attention],
                               mode='dot', dot_axes=(1, 1)))


# print(Model(x, hidden_state).predict(X_batch))

def model(train):
    def decoder_unit():
        return GRU(hidden_dim,
                   return_sequences=True,
                   input_shape=(1, hidden_dim,))

    expanded_h = Input(shape=(1, hidden_dim), dtype='int32', name='expanded_h')
    decode, reencode = Sequential(), Sequential()
    for _ in range(depth):
        decode.add(decoder_unit())
        reencode.add(decoder_unit())

    pred = decode(expanded_h)
    state = reencode(pred)
    output = [pred, state]
    decode = feedback(Model(input=[expanded_h], output=output),
                      output_length=timesteps_o,
                      num_outputs=len(output),
                      unroll=False,
                      feedback_function=lambda t: t[1])
    output = decode(hidden_state)
    return Model(input=[x], output=output)


train_model = model(train=True)
test_model = model(train=False)
train_model.compile(loss='binary_crossentropy',
                    optimizer='sgd')
print(train_model.predict([X_batch]))  # , Y_batch]))
# train_model.train_on_batch(X_batch, Y_batch)
