import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import datetime
from string import Formatter


class Profiler(object):

    def __enter__(self):
        self._start = datetime.datetime.now()

    def __exit__(self, type, value, traceback):
        self._end = datetime.datetime.now()
        td = self._end - self._start
        print("Elapsed time: {}".format(strfdelta(td, fmt='{H:02}h {M:02}m {S:02}s')))


class Logger(object):

    def __init__(self, logfile):
        self.logfile = logfile

    def write(self, line):
        with open(self.logfile, 'a') as f:
            f.write(line + os.linesep)


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta) * 60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta) * 3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta) * 86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta) * 604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def set_seed(seed=None, cuda=False, verbose=False):
    if seed is not None:
        if verbose: print("Set seed: ", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def show_results(losses, results, inputs, outputs, xs, titles, losstitle="K-shot regression", xlim=None, ylim=None):
    # if only one result
    if isinstance(results, dict):
        losses = [losses]
        results = [results]
        titles = [titles]

    assert len(losses) == len(results) == len(inputs) == len(outputs) == len(xs) == len(titles), \
        "Number of losses and results are different: {} =/= {} =/= {} =/= {} =/= {} =/= {}" \
            .format(len(losses), len(results), len(inputs), len(outputs), len(xs), len(titles))

    f, axs = plt.subplots(nrows=2, ncols=len(losses), figsize=(10 * len(losses), 15), sharex=False)
    ax0 = axs[0, 0] if len(axs.shape) > 1 else axs[0]
    gs = ax0.get_gridspec()
    # remove the underlying axes
    if len(axs.shape) > 1:
        for ax in axs[1, :]:
            ax.remove()
    else:
        axs[1].remove()
    axloss = f.add_subplot(gs[1:, :])
    f.tight_layout(h_pad=8.0)

    params = {
        "pre-update": {
            "key": "pre-update",
            "color": "green",
            "linestyle": ":"
        },
        1: {
            "key": "1 grad",
            "color": "green",
            "linestyle": "-."
        },
        10: {
            "key": "10 grads",
            "color": "green",
            "linestyle": "--"
        },
        "ground truth": {
            "key": "ground truth",
            "color": "red",
            "linestyle": "-"
        }
    }

    # calculate optimal limits
    if xlim is None:
        xlim = (-6, 6)
    if ylim is None:
        allys = []
        for i in range(len(losses)):
            for key, ys in results[i].items():
                if key in params.keys():
                    allys.append(ys)
        ys = np.concatenate(allys).ravel()
        minys, maxys = np.min(ys), np.max(ys)
        ylim = (1.05 * minys, 1.05 * maxys)

    for i in range(len(losses)):

        ax0 = axs[0, i] if len(axs.shape) > 1 else axs[0]
        colors = ["green", "blue"]

        # PLOT GRAPHS
        for key, ys in results[i].items():
            if key in params.keys():
                ax0.plot(xs[i], ys,
                         color=params[key]["color"],
                         linestyle=params[key]["linestyle"],
                         label=params[key]["key"])

        # add datapoints
        ax0.plot(inputs, outputs, marker="^", markersize=10, markerfacecolor="blue", linestyle="")

        # title and legend
        ax0.axes.set_title(titles[i], fontsize=22)
        ax0.axes.set_xlim(xlim)
        ax0.axes.set_ylim(ylim)
        ax0.legend(loc='upper center',
                   fontsize='x-large',
                   bbox_to_anchor=(0.5, -0.1),
                   ncol=4)

        # PLOT LOSS
        axloss.plot(np.arange(len(losses[i])), losses[i], marker="o", markersize=10, color=colors[i], label=titles[i])

    # title, axis names, legend
    axloss.axes.set_title(losstitle, fontsize=22)
    axloss.axes.set_xlabel("number of gradient steps", fontsize=22)
    axloss.axes.set_ylabel("mean squared error", fontsize=22)
    axloss.legend(loc='upper right',
                  fontsize='x-large',
                  ncol=1)

    plt.show()
