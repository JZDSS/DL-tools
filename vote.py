import os
import argparse
import numpy as np

FLAGS = None


def main():

    meta = {}
    f = open(os.path.join(FLAGS.meta_dir, 'spc_classes.txt'), 'r')
    line = f.readline()
    while line:
        label, class_name = line.split(' ')
        class_name = class_name[0:-1]
        # meta[int(label)] = class_name
        meta[class_name] = int(label)
        line = f.readline()
    f.close()
    truth = np.load('truth.npy')
    preds = os.listdir(FLAGS.dir)

    res = np.zeros(shape=[2750*9, 10])
    acc = []
    for file in preds:
        curr = []
        f = open(os.path.join(FLAGS.dir, file))
        f.readline()
        n = 0
        while line:
            l = f.readline().split(',')[1].split('\n')[0]
            res[n, meta[l]] += 1
            n = n + 1
            curr.append(meta[l])
        acc.append(np.mean(curr == truth))

    voted = np.argmax(res, axis=2)
    acc.append(np.mean(voted == truth))
    print(acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        default='./predictions/',
        help='Directory where all prediction files are stored'
    )
    parser.add_argument(
        '--meta_dir',
        type=str,
        default='/data/qiyao/official/meta',
        help='Meta directory'
    )
    FLAGS, _ = parser.parse_known_args()
    main()