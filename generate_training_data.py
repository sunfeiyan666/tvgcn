# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import __future__

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, scaler=None
):
    """
    data:total flow, average speed, and average occupancy
    Generate samples from
    :param df: (l,n)->(l,n,,f)
    :param x_offsets:
    :param y_offsets:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    #num_samples, num_nodes = df.shape
    num_samples, num_nodes, num_features = df.shape
    print(df.shape)
    #print(df[16991,0,0])
    #data = np.expand_dims(df.values, axis=-1) #(l,n,1)
    feature_list = [df]
    #df = df.tolist()

    data = np.concatenate(feature_list, axis=-1)  #(l,n,f)
    print("data shape: ", data.shape)
    x, y = [], []
    min_t = abs(min(x_offsets))  #11
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        #print("x shape: ", x.shape, ", y shape: ", y.shape)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    return x, y  #(l,t,n,f)


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = np.load(args.filename)['arr_0']
    #df = pd.read_hdf(args.traffic_df_filename) #(length, num of nodes)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))  #[-11, -10,...,0]
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1)) #[1,2,...,11,12]
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)  #(l,t,n,f)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/PEMS04/", help="Output directory.")
    parser.add_argument("--filename", type=str, default="data/PEMS04/pems04.npz", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
