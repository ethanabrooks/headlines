from __future__ import print_function

import datetime
import theano
import theano.tensor as T
import numpy as np
import time
import sys

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return ":".join((str(int(t)) for t in (hours, minutes, seconds)))

def print_progress(epoch, instances_processed, num_instances, loss, start_time):

    progress = round(float(instances_processed) / num_instances, ndigits=3)
    elapsed_time = time.time() - start_time
    eta = elapsed_time / progress
    print(eta)
    print(format_time(eta))
    elapsed_time, eta = map(format_time, (elapsed_time, eta))
    print('\r###\t{:<10d}{:<10.1%}{:<10.5f}{:<10}{:<10}###'
          .format(epoch, progress, float(loss), elapsed_time, eta), end='')
    sys.stdout.flush()


print_progress(1, 200, 2000, 12, time.time())
