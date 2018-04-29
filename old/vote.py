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

    truth = []
    for name in os.listdir('/data/qiyao/valid_imgs'):
        for key in meta:
            if key in name:
                truth.append(meta[key])
                break

    truth = np.array(truth)    # truth = np.load('truth.npy')
    preds = os.listdir(FLAGS.dir)
    print(preds)
    res = np.zeros(shape=[5040, 10], dtype=np.float32)
    acc = []
    for file in preds:
        tmp = np.zeros(shape=[5040, 10], dtype=np.float32)
        curr = []
        f = open(os.path.join(FLAGS.dir, file))
        line = f.readline()
        n = 0
        while line:
            line = f.readline()
            if not ',' in line:
                break
            l = line.split(',')[1].split('\n')[0]
            tmp[n, meta[l]] += 1
            n = n + 1
            curr.append(meta[l])
        acc.append(np.mean(curr == truth))
        res += tmp
    voted = np.argmax(res, axis=1)
    acc.append(np.mean(voted == truth))
    print(acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        default='/home/amax/QiYao/multi-gpus/official/valid',
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