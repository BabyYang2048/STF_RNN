from utils import *
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STF-RNN: Space Time Features-based Recurrent Neural Network for Predicting People Next Location')
    parser.add_argument('-e1',  dest='emb_size1', type=int, default=100, help='The space embedding size.')
    parser.add_argument('-e2',  dest='emb_size2', type=int, default=6, help='The time embedding size.')
    parser.add_argument('-r',  dest='rnn_size', type=int, default=20, help='The recurrent hidden size.')
    parser.add_argument('-w',  dest='window_size', type=int, default=2, help='The window size.')
    parser.add_argument('-t',  dest='min_time', type=int, default=1800, help='The time threshold in seconds.')
    parser.add_argument('-d',  dest='min_dist', type=float, default=0.2, help='The distance threshold.')
    parser.add_argument('-e',  dest='epochs', type=int, default=100, help='The number of training epochs.')
    parser.add_argument('-b',  dest='batch_size', type=int, default=30, help='The mini batch size.')

    args = parser.parse_args()
    print(args)

    acc = []
    window_size = args.window_size
    k = 0

    # test
    model_id = [0, 101, 126, 14, 23, 3, 38, 50, 51, 6, 66, 85, 89, 91]
    for user_id in model_id:
        print("Processing user %d." % user_id)
        try:
            data_path = 'Data/%03d/Trajectory/*.plt' % user_id
            traj = read_data(data_path)
            # print('read_data')

            s_inputs, t_inputs, outputs, N = \
                prpare_data(traj, window_size=window_size, min_dist=args.min_dist, min_time=args.min_time)

            if N == 0 or len(s_inputs) <= 8:
                continue
            outputs_ = to_categorical(outputs, N)
            s_inputs = np.array(s_inputs).astype(np.int64)
            t_inputs = np.array(t_inputs).astype(np.int64)

            model_path = 'stf_rnn_save_models/model_u%d.h5' % user_id
            # print('model_path=',model_path)
            model = load_model(model_path)
            # print(model.summary())
            
            loss, accuracy = model.evaluate([s_inputs, t_inputs], outputs_)
            print('acc=', str(accuracy))
            acc.append(accuracy)
            k += 1
        except :
            pass
            # print("here is pass")
       
    print(np.mean(acc))
    # print(acc)



