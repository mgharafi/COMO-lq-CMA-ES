#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A short and simple example experiment with restarts.

The script is fully functional but also emphasises on readability. It
features restarts, timings and recording termination conditions.

To benchmark a different solver, `fmin` must be re-assigned and another
`elif` block added around line 119 to account for the solver-specific
call.

When calling the script, previously assigned variables can be re-assigned
via a ``name=value`` argument without white spaces, where ``value`` is
interpreted as a single python literal. Additionally, ``batch`` is recognized
as argument defining the `current_batch` number and the number of `batches`,
like ``batch=2/8`` runs batch 2 of 8.

Examples, preceeded by "python" in an OS shell and by "run" in an IPython
shell::

    example_experiment2.py budget_multiplier=3  # times dimension

    example_experiment2.py budget_multiplier=1e4 cocopp=None  # omit post-processing
    
    example_experiment2.py budget_multiplier=1e4 suite_name=bbob-biobj

    example_experiment2.py budget_multiplier=1000 batch=1/16

Post-processing with `cocopp` is only invoked in the single-batch case.

Details: ``batch=9/8`` is equivalent to ``batch=1/8``. The first number
is taken modulo to the second.

See the code: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment2.py>`_

See a beginners example experiment: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment_for_beginners.py>`_

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
import sys
import time  # output some timings per evaluation
from collections import defaultdict
from cma.fitness_models import kendall_tau
import os # to show post-processed results in the browser
from pprint import pprint
import numpy as np  # for median, zeros, random, asarray
import cocoex  # experimentation module
import copy
try: import cocopp  # post-processing module
except: pass

### MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    try: import mkl
    except ImportError: disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting mkl threads num to", nt)

if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)

### solver imports (add other imports if necessary)
sys.path.append("/home/randopt/mgharafi/mutual_archive_experiments/coco_tests/coco_tests/home/mutual_archive")
from lqcomo import LQCOMO

suite_name = "bbob-biobj"  # see cocoex.known_suite_names
budget_multiplier = 2  # times dimension, increase to 10, 100, ...
suite_filter_options = (""  # without filtering, a suite has instance_indices 1-15
                        "dimensions: 2,3,5,10,20 "  # skip dimension 40
                        "instance_indices: 1-5 "  # relative to suite instances
                       )
# for more suite filter options see http://numbbo.github.io/coco-doc/C/#suite-parameters
suite_year_option = "year: 2022"  # determine instances by year, not all years work for all suites :-(

batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant


# hv_all = False # HV-based on the whole archive if True
model = True # If True the surrogate is used
number_of_kernels = 10
prefix = ''
popsize_factor = 1

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(), {'batch': 'current_batch/batches'}, print=print)
    globals().update(input_params)  # (re-)assign variables
# print(model, popsize_factor, hv_all, number_of_kernels)

model = bool(model)
popsize_factor = int(popsize_factor)
number_of_kernels = int(number_of_kernels)
prefix = str(prefix)

model_name = 'LQ' if model else 'Como'
# prefix_ = 'All' if hv_all else 'ker'

output_folder = f'{prefix}/P{popsize_factor}K{number_of_kernels}B{budget_multiplier}/{model_name}/'#'lq-como-32/'
output_folder2 = f'{prefix}/P{popsize_factor}K{number_of_kernels}B{budget_multiplier}/ker{model_name}/'#'lq-como-32/'

# extend output folder input parameter, comment out if desired otherwise
output_folder += '%dD_on_%s' % (int(budget_multiplier), suite_name)
output_folder2 += '%dD_on_%s' % (int(budget_multiplier), suite_name)

if batches > 1:
    output_folder += "_batch%03dof%d" % (current_batch, batches)
    output_folder2 += "_batch%03dof%d" % (current_batch, batches)

### prepare
suite = cocoex.Suite(suite_name, suite_year_option, suite_filter_options)
suite2 = cocoex.Suite(suite_name, suite_year_option, suite_filter_options)

observer_options = ("result_folder: " + output_folder + " log_nondominated: none")
observer2_options = ("result_folder: " + output_folder2 + " log_nondominated: none")

observer = cocoex.Observer(suite_name, observer_options)
observer2 = cocoex.Observer(suite_name, observer2_options)

minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

def kendallBasedUpdate(lq, threshold=.6):
    def predicat(kernel):
        if kernel.countiter <= 0:
            return True
        off = lq.solver.offspring[0][1]
        # print(off)
        return kendall_tau(
                [kernel.surrogate.model.eval(x) for x in off],
                [lq.uhvi(kernel=kernel, forget=True)(x) for x in off]
                ) >= threshold
    return predicat

def reeval(optimizer, problem):
    def wrapper():
        for k in optimizer.solver._told_indices:
            if optimizer.solver.surrogate:
                evaluated = len([_ for _ in optimizer.solver.kernels[k].surrogate.evals.evaluated if _]) + 1
                for _ in range(evaluated):
                    problem(optimizer.solver.kernels[k].incumbent)
            else:
                for _ in range(optimizer.solver.kernels[k].popsize + 1):
                    problem(optimizer.solver.kernels[k].incumbent)
    return wrapper

def kendallBasedUpdate(threshold=.6):
    def predicat(kernel):
        if kernel.countiter <= 0:
            return True
        off = [v['pheno'].tolist() for k, v in kernel.sent_solutions.items()]
        # print(off)
        return kendall_tau(
                [kernel.surrogate.model.eval(x) for x in off],
                [kernel.uhvi(x) for x in off]
                ) < threshold
    return predicat

def skipNIterations(numberOfSkippedIterations = 1):
    def predicat(kernel):
        return kernel.countiter % (numberOfSkippedIterations + 1) == 1 or not kernel.countiter > 1
    return predicat

UPDATE_POLICIES = {
    'default' : lambda _ : True,
    'skip-one' : skipNIterations,
    'kendall-based-t60' : kendallBasedUpdate,
}

### go
print('*** benchmarking %s from %s on suite %s ***'
      % ("lq-como", "v1", suite_name))
time0 = time.time()
for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings)[-2],
                 np.median(timings[sorted(timings)[-2]])), end='')
    fproblem = suite2.get_problem(problem.id)

    fproblem.observe_with(observer2) 
    problem.observe_with(observer)  # generate the data for cocopp post-processing

    problem(np.zeros(problem.dimension))  # making algorithms more comparable
    propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
    evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                            max((problem.evaluations, problem.evaluations_constraints)))
    time1 = time.time()
    
    irestart = 0

    lq = LQCOMO(
        X0 = np.random.uniform(-5, 5, (number_of_kernels, problem.dimension)),
        sigma0 = 2/np.sqrt(problem.dimension), #Too large (2 / sqrt(d))
        fitness = problem,
        reference_point= problem.largest_fvalues_of_interest,
        inopts={
            'popsize_factor': popsize_factor,
            'tolfun': '0e-11  #v termination criterion: tolerance in function value, quite useful',
            'tolfunhist': '0e-12  #v termination criterion: tolerance in function value history',
            'tolstagnation': '0 * int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations',
            'tolx': '0e-11  #v termination criterion: tolerance in x-changes'
            },
        use_surrogate=model,
        tau_tresh=.85,
        UPDATE_POLICIES=UPDATE_POLICIES,
        min_evals_percent=0,
        return_true_fitnesses=True,
        )

    callback = reeval(lq, fproblem)

    lq.optimize(
        user_stop=lambda : lq.solver.countevals > int(problem.dimension * budget_multiplier),
        callback = callback,
        user_updates_rules='default',
        )

    stoppings[problem.index].append(lq.solver.termination_status)

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)
    # with open('exdata/' + output_folder + '_stopping_conditions.pydict', 'wt') as file_:
    #     file_.write("# code to read in these data:\n"
    #                 "# import ast\n"
    #                 "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
    #                 "#     stoppings = ast.literal_eval(file_.read())\n"
    #                 % output_folder)
    #     file_.write(repr(dict(stoppings)))

### print timings and final message
print("\n   %s %d-D done in %.1e seconds/evaluations"
      % (minimal_print.stime, sorted(timings)[-1], np.median(timings[sorted(timings)[-1]])))
if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

print("Timing summary:\n"
      "  dimension  median seconds/evaluations\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")

### post-process data
# if batches == 1 and 'cocopp' in globals() and cocopp not in (None, 'None'):
#     cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
#     webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
