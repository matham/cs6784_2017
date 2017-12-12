import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import re

root = r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-binary'
train = [f for f in os.listdir(root) if f.startswith('train') and f.endswith('.csv')]
test = [f for f in os.listdir(root) if f.startswith('test_ft') and f.endswith('.csv')]
layer_order = ['0']
for block in range(1, 4):
    for layer in range(1, 17):
        layer_order.append('{}_{}'.format(block, layer))
    layer_order.append(str(block))


convert = lambda text: int(text) if text.isdigit() else text.lower()
alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]


def init_plotting():
    #plt.rcParams['figure.figsize'] = (8, 3)
    plt.rcParams['font.size'] = 15
    # plt.rcParams['font.family'] = 'T'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    #print(plt.rcParams['savefig.dpi'])
    #plt.rcParams['savefig.dpi'] = 2* plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['axes.linewidth'] = 3.


init_plotting()


def finish_plot():
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')


def get_files(root, start_name):
    return [os.path.join(root, f) for f in os.listdir(root) if
            f.startswith(start_name) and f.endswith('.csv')]


def running_mean(x, n):
    l = n // 2
    r = n - l
    x = np.concatenate((np.ones(l) * x[0], x, np.ones(r) * x[-1]))
    cumsum = np.cumsum(x)
    return (cumsum[n:] - cumsum[:-n]) / n


def load_layer_files(files, other_files=None):
    results = {}
    for f in files:
        epoch, loss, err = [], [], []
        name = os.path.basename(f)
        name = name.replace('test_ft=[', '').replace('train_ft=[', '').replace('].csv', '')
        name = re.match('.*?([0-9]+)(?:=([0-9]+)|$)', name).groups()
        name = name[0] if name[1] is None else '{}_{}'.format(*name)
        results[name] = epoch, loss, err
        with open(f, 'r') as fh:
            for row in csv.reader(fh):
                epoch.append(float(row[0]))
                loss.append(float(row[1]))
                err.append(float(row[2]))

    if other_files:
        results = {}
        for f1, f2 in zip(files, other_files):
            epoch, loss, err = [], [], []
            name = os.path.basename(f2)
            name = name.replace('test_ft=[', '').replace('train_ft=[', '').replace('].csv', '')
            name = re.match('.*?([0-9]+)(?:=([0-9]+)|$)', name).groups()
            name = name[0] if name[1] is None else '{}_{}'.format(*name)
            results[name] = epoch, loss, err
            with open(f1, 'r') as fh1:
                with open(f2, 'r') as fh2:
                    for row1, row2 in zip(csv.reader(fh1), csv.reader(fh2)):
                        epoch.append(float(row1[0]))
                        loss.append((float(row1[1]) + float(row2[1])) / 2.)
                        err.append((float(row1[2]) + float(row2[2])) / 2.)
    return results


def plot_layer_files(run, measure=2, label='Test Error', smooth=0):
    for f, cols in sorted(run.items(), key=lambda x: layer_order.index(x[0])):
        epoch, col = cols[0], cols[measure]
        if smooth:
            col = running_mean(col, smooth)

        plt.plot(epoch, col, label=f)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1)
    plt.gcf().subplots_adjust(right=0.77)
    #plt.tight_layout()
    plt.show()


def plot_layer_files_summary(runs, bin_runs=[], measure=2, label='Test Error', ax=None):
    if ax is None:
        ax = plt.subplot()

    colors = []
    multi = len(runs) > 1
    for i, run in enumerate(runs):
        data = list(sorted(run.items(), key=lambda x: layer_order.index(x[0])))
        labels = [d[0] for d in data]
        group = 'Base' if bin_runs else 'Epoch'
        group += str(i) if multi else ''
        for e in (0, len(data[0][1][0]) // 2, len(data[0][1][0]) - 1):
            p = ax.plot(range(len(labels)), [d[1][measure][e] for d in data], 'x--',
                        label='{} ({})'.format(group, e + 1))
            colors.append(p[0].get_color())

    if bin_runs:
        j = 0
        for i, bin_run in enumerate(bin_runs):
            data = list(sorted(bin_run.items(), key=lambda x: layer_order.index(x[0])))
            idx = str(i) if multi else ''
            for e in (0, len(data[0][1][0]) // 2, len(data[0][1][0]) - 1):
                ax.plot(range(len(labels)), [d[1][measure][e] for d in data], '.--',
                        label='Bin{} ({})'.format(idx, e + 1), color=colors[j])
                j += 1

    plt.xticks(np.arange(len(labels)) + .22)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel(label)
    plt.legend(ncol=3)
    #plt.gcf().subplots_adjust(right=0.77)
    plt.tight_layout()
    plt.show()


def plot_a_b_bin_baseline(baseline_root, bin_root, measure=3, label='Test Error', stage='test', ax=None):
    if ax is None:
        ax = plt.subplot()
    labels = ['A', 'B100', 'B40']

    colors = []
    for i, (item, smooth) in enumerate([('1', 15), ('_ft=[all]', 15), ('_ft=[0]', 4)]):
        if stage == 'train':
            smooth *= 400
        with open(os.path.join(baseline_root, stage + item + '.csv'), 'r') as fh:
            epoch, col = [], []
            for row in csv.reader(fh):
                epoch.append(float(row[0]))
                col.append(float(row[measure]))
        col = running_mean(col, smooth)
        p = ax.plot(epoch, col, 'x--', label='Baseline {}'.format(labels[i]))
        colors.append(p[0].get_color())

    for i, (item, smooth) in enumerate([('1', 15), ('_ft=[all]', 15), ('_ft=[0]', 4)]):
        if stage == 'train':
            smooth *= 400
        with open(os.path.join(bin_root, stage + item + '.csv'), 'r') as fh:
            epoch, col = [], []
            for row in csv.reader(fh):
                epoch.append(float(row[0]))
                col.append(float(row[measure]))
        col = running_mean(col, smooth)
        ax.plot(epoch, col, '.--', label='Binary {}'.format(labels[i]), color=colors[i])

    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend(loc='upper right', ncol=1)
    plt.tight_layout()
    plt.show()


def plot_60_bin_baseline(baseline_root, baseline_root2, bin_root, bin_root2, measure=2, label='Test Error'):
    ax = plt.subplot()

    res = []
    for root in (baseline_root, baseline_root2, bin_root, bin_root2):
        with open(os.path.join(root, 'test_ft=[all].csv'), 'r') as fh:
            epoch, col = [], []
            for row in csv.reader(fh):
                epoch.append(float(row[0]))
                col.append(float(row[measure]))
        res.append(col)

    base1, base2, bin1, bin2 = res
    p = ax.plot(epoch, base1, 'x--', label='Baseline 1')
    ax.plot(epoch, bin1, '.--', label='Binary 1', color=p[0].get_color())
    p = ax.plot(epoch, base2, 'x--', label='Baseline 2')
    ax.plot(epoch, bin2, '.--', label='Binary 2', color=p[0].get_color())

    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend(loc='upper right', ncol=1)
    plt.tight_layout()
    plt.show()


def plot_wrn():
    with open(r'G:\Python\cs6784fa17\paper\wrn_B.csv', 'r') as fh:
        epoch, base1, bin1, base2, bin2 = [], [], [], [], []
        first = True
        for row in csv.reader(fh):
            if first:
                first = False
                continue
            epoch.append(float(row[0]))
            base1.append(float(row[1]))
            bin1.append(float(row[2]))
            base2.append(float(row[3]))
            bin2.append(float(row[4]))

    ax = plt.subplot()
    p = ax.plot(epoch, base1, 'x--', label='Baseline 1')
    ax.plot(epoch, bin1, '.--', label='Binary 1', color=p[0].get_color())
    p = ax.plot(epoch, base2, 'x--', label='Baseline 2')
    ax.plot(epoch, bin2, '.--', label='Binary 2', color=p[0].get_color())

    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.legend(loc='upper right', ncol=1)
    plt.tight_layout()
    plt.show()


def plot_imagenet():
    with open(r'G:\Python\cs6784fa17\paper\imagenet.csv', 'r') as fh:
        epoch, base, bin = [], [], []
        first = True
        for row in csv.reader(fh):
            if first:
                first = False
                continue
            epoch.append(float(row[0]))
            base.append(float(row[1]))
            bin.append(float(row[2]))

    ax = plt.subplot()
    p = ax.plot(epoch, base, '.--', label='Baseline')
    ax.plot(epoch, bin, '.--', label='Binary')

    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.legend(loc='upper right', ncol=1)
    plt.tight_layout()
    plt.show()


def plot_b_weight(fname):
    with open(os.path.join(r'G:\Python\cs6784fa17\paper', fname), 'r') as fh:
        header = None
        for row in csv.reader(fh):
            if not header:
                header = row[1:]
                continue
    values = list(map(float, row[1:]))

    ax = plt.subplot()
    ax.bar(range(len(values)), values, color='b')

    plt.xticks(np.arange(len(values)))
    ax.plot([-.5, len(values) - .5], [min(values), min(values)], 'r--')
    ax.set_xticklabels(header)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Test Error')
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bin_baseline(fname, fname_b):
    with open(os.path.join(r'G:\Python\cs6784fa17\paper', fname), 'r') as fh:
        header = None
        for row in csv.reader(fh):
            if not header:
                header = row[1:]
                continue
    values = list(map(float, row[1:]))
    base = values[0::2]
    bin = values[1::2]

    with open(os.path.join(r'G:\Python\cs6784fa17\paper', fname_b), 'r') as fh:
        header = None
        for row in csv.reader(fh):
            if not header:
                header = row[1:]
                continue
    values = list(map(float, row[1:]))
    baseb = values[0::2]
    binb = values[1::2]

    ax = plt.subplot()
    ax.scatter(base, bin, label='A')
    ax.scatter(baseb, binb, label='B')
    y1, y2 = ax.get_ylim()
    x1, x2 = ax.get_xlim()
    ax.plot([0, 100], [0, 100], 'g--')
    ax.set_xlim([min(y1, x1), max(y2, x2)])
    ax.set_ylim([min(y1, x1), max(y2, x2)])

    ax.set_xlabel('Baseline')
    ax.set_ylabel('Binary (B=.4)')
    ax.set_title('Test Error')
    plt.legend()
    plt.tight_layout()
    plt.show()


def long_tail(test1, test2, measure=2, label='Test Error'):
    base1, bin1 = [], []
    for d in os.listdir(test1):
        with open(os.path.join(test1, d, 'test_ft=[all].csv'), 'r') as fh:
            header = None
            vals = []
            for row in csv.reader(fh):
                if not header:
                    header = row[1:]
                    continue
                vals.append(float(row[measure]))
        if d.startswith('base'):
            base1.append(vals)
        else:
            bin1.append(vals)
    base2, bin2 = [], []
    for d in os.listdir(test2):
        with open(os.path.join(test2, d, 'test_ft=[all].csv'), 'r') as fh:
            header = None
            vals = []
            for row in csv.reader(fh):
                if not header:
                    header = row[1:]
                    continue
                vals.append(float(row[measure]))
        if d.startswith('base'):
            base2.append(vals)
        else:
            bin2.append(vals)
    base1 = np.array(base1)
    base2 = np.array(base2)
    bin1 = np.array(bin1)
    bin2 = np.array(bin2)

    base1 = np.mean(base1, axis=0)
    base2 = np.mean(base2, axis=0)
    bin1 = np.mean(bin1, axis=0)
    bin2 = np.mean(bin2, axis=0)
    print(base1[-1], bin1[-1], base2[-1], bin2[-1])

    ax = plt.subplot()
    p = ax.plot(base1, 'x--', label='Baseline 1')
    ax.plot(bin1, '.--', label='Binary 1', color=p[0].get_color())
    p = ax.plot(base2, 'x--', label='Baseline 2')
    ax.plot(bin2, '.--', label='Binary 2', color=p[0].get_color())

    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.legend(loc='upper right', ncol=1)
    plt.tight_layout()
    plt.show()


def plot_binary(fname, sort_by_intensity=False):
    results = []
    with open(fname, 'r') as fh:
        for row in csv.reader(fh):
            if not results:
                for col in row:
                    results.append([col])
            else:
                for i, col in enumerate(row):
                    results[i].append(col)

    for col in sorted(results[1:], key=lambda x: x[0]):
        plt.plot(results[0][1:], col[1:], label=col[0])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

    ax = plt.subplot()
    values = []
    labels = []
    for col in sorted(results[1:], key=lambda x: x[0]):
        values.append(min(col[1:]))
        labels.append(col[0])

    if sort_by_intensity:
        labels = sorted(labels, key=lambda x: values[labels.index(x)])
        values = sorted(values)
    ax.bar(range(len(values)), values, color='b')
    plt.xticks(np.arange(len(values)))
    ax.plot([-.5, len(values) - .5], [min(values), min(values)], 'r--')
    ax.set_xticklabels(labels)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Minimum Error')
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot2(baseline, binary, label1='Baseline', label2='B=0.4'):
    base_err = []
    bin_err = []
    with open(baseline, 'r') as fh:
        for row in csv.reader(fh):
            base_err.append(row[-1])
    with open(binary, 'r') as fh:
        for row in csv.reader(fh):
            bin_err.append(row[-1])
    plt.plot(base_err, label=label1)
    plt.plot(bin_err, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

# layers0 = load_layer_files(get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-transfer-finetune-layers100', 'test_ft'))
# plot_layer_files(layers0, smooth=15)
# plot_layer_files_summary([layers0])


# base = load_layer_files(
#     get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-baseline', 'test_ft'),
#     get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-baseline2', 'test_ft')
# )
# binary = load_layer_files(
#     get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-binary', 'test_ft'),
#     get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-binary2', 'test_ft')
# )
# base2 = load_layer_files(get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-baseline2', 'test_ft'))
# binary2 = load_layer_files(get_files(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-freeze-binary2', 'test_ft'))
#
# plot_layer_files(base, smooth=4)
# plot_layer_files(binary, smooth=4)
# plot_layer_files_summary([base], [binary])


# plot_a_b_bin_baseline(
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-baseline5',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-transfer-binary-classifier14')
# plot_a_b_bin_baseline(
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-baseline5',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-transfer-binary-classifier14',
#     label='Train Error', stage='train')
# plot_a_b_bin_baseline(
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-baseline5',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-transfer-binary-classifier14',
#     label='Test Loss', stage='test', measure=1)
# plot_a_b_bin_baseline(
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-baseline5',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-transfer-binary-classifier14',
#     label='Train Loss', stage='train', measure=1)


# plot_60_bin_baseline(
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-60-baseline',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-60-baseline2',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-60-binary',
#     r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-60-binary2')


# plot_wrn()
# plot_imagenet()
# plot_b_weight('b_weights.csv')
# plot_b_weight('b_weights_other.csv')
# plot_bin_baseline('baseline vs binary A.csv', 'baseline vs binary.csv')
long_tail(r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-60_13', r'G:\Python\cs6784fa17\cs6784_2017-master\saved_results\cifar100-60_14')
#plot_binary(r'G:\Python\6784\test_ft_other.csv', True)
#plot2(r'G:\Python\6784\results\test_ft=[all].csv', r'G:\Python\6784\test_ft=image.csv')
#plot2(r'G:\Python\6784\cifar100-transfer-finetune-blocks2\test1.csv', r'G:\Python\6784\cifar100-transfer-binary-classifier9\test1.csv')
