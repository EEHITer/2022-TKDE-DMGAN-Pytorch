from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import numpy as np

def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param data:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, features = data.shape
    # data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_samples(args):
    graph_signal_matrix_filename = args.traffic_filename

    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'data' in keys:
        x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
        )
        # Predict the next one hour
        y_offsets = np.sort(np.arange(1, 13, 1))
        x, y = generate_graph_seq2seq_io_data(
            data['data'],
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            add_time_in_day=True,
            add_day_in_week=False,
        )
        print("x shape: ", x.shape, ", y shape: ", y.shape)

        num_samples = x.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.6)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = (
            x[num_train: num_train + num_val],
            y[num_train: num_train + num_val],
        )
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]

        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
                os.path.join(args.output_dir, "%s.npz" % cat),
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/PEMS07/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_filename",
        type=str,
        default="data/PEMS08/PEMS07.npz",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    generate_samples(args)