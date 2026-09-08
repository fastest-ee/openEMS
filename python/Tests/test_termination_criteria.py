# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Termination of the FDTD loop: when it stops, and why.

The end criteria is evaluated every EndCriteriaCheckInterval timesteps. It used
to be evaluated on a 4 second wall-clock period instead, which made the number
of simulated timesteps -- and therefore the frequency resolution df = 1/(N*dt)
of every result -- depend on how fast the machine was.
"""

import os
import tempfile
import time
import unittest

import numpy as np

from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS

SLOW = os.environ.get('OPENEMS_SLOW_TESTS', '') not in ('', '0')


def _make_sim(interval=None, n=21, half=10.0, f0=6e9, fc=3e9, **kw):
    """A small air box with a short lumped port in the middle, absorbed by MUR
    on all sides. The excitation carries no DC component, so the stored energy
    keeps decaying once the pulse is over and the end criteria is reached after
    a few hundred timesteps.

    Lowering f0/fc raises the Nyquist rate in timesteps, and with it the number
    of timesteps the loop advances per iteration -- which is what decides
    whether a given check interval is longer or shorter than one iteration."""
    fdtd = openEMS(**kw)
    if interval is not None:
        fdtd.SetEndCriteriaCheckInterval(interval)
    fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(['MUR'] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    for direction in 'xyz':
        grid.SetLines(direction, np.linspace(-half, half, n))
    fdtd.AddLumpedPort(1, 50, [0, 0, -2], [0, 0, 2], 'z', excite=1)
    return fdtd


def _run(fdtd, tag, **kw):
    path = os.path.join(tempfile.gettempdir(), 'test_termcrit_' + tag)
    t0 = time.time()
    fdtd.Run(path, cleanup=True, verbose=0, dump_statistics=True, **kw)
    wall = time.time() - t0
    with open(os.path.join(path, 'openEMS_stats.txt')) as fh:
        vals = [line.split('\t')[0].strip() for line in fh]
    return {'timesteps': int(float(vals[2])), 'reason_code': int(vals[6]), 'wall': wall}


class Test_TerminationReason(unittest.TestCase):
    """converged / max_timesteps / max_run_time must not collapse into one status:
    a run that was cut short has to be distinguishable from one that converged."""

    def test_converged(self):
        fdtd = _make_sim(interval=64, NrTS=50000, EndCriteria=1e-6)
        st = _run(fdtd, 'converged')
        self.assertEqual(fdtd.GetTerminationReasonString(), 'converged')
        self.assertEqual(fdtd.GetTerminationReason(), 1)
        self.assertEqual(st['reason_code'], 1)
        self.assertLess(st['timesteps'], 50000)

    def test_max_timesteps(self):
        fdtd = _make_sim(NrTS=500, EndCriteria=1e-30)
        st = _run(fdtd, 'maxts')
        self.assertEqual(fdtd.GetTerminationReasonString(), 'max_timesteps')
        self.assertEqual(fdtd.GetTerminationReason(), 2)
        self.assertEqual(st['reason_code'], 2)
        self.assertEqual(st['timesteps'], 500)

    def test_max_run_time(self):
        fdtd = _make_sim(NrTS=10 ** 8, EndCriteria=1e-30, MaxRunTime=2)
        st = _run(fdtd, 'maxruntime')
        self.assertEqual(fdtd.GetTerminationReasonString(), 'max_run_time')
        self.assertEqual(fdtd.GetTerminationReason(), 3)
        self.assertEqual(st['reason_code'], 3)
        # the limit is checked once per loop iteration, so it is honoured to
        # within one chunk of timesteps, not to the microsecond
        self.assertGreaterEqual(st['wall'], 2.0)
        self.assertLess(st['wall'], 60.0)
        self.assertLess(st['timesteps'], 10 ** 8)

    def test_all_three_codes_differ(self):
        codes = set()
        for kw, tag in (({'NrTS': 50000, 'EndCriteria': 1e-6, 'interval': 64}, 'd_conv'),
                        ({'NrTS': 500, 'EndCriteria': 1e-30}, 'd_ts'),
                        ({'NrTS': 10 ** 8, 'EndCriteria': 1e-30, 'MaxRunTime': 2}, 'd_rt')):
            fdtd = _make_sim(**kw)
            _run(fdtd, tag)
            codes.add(fdtd.GetTerminationReason())
        self.assertEqual(len(codes), 3)

    def test_convergence_on_the_last_timestep_is_reported_as_converged(self):
        # A run that reaches the end criteria on the very timestep the max.
        # number of timesteps is reached has converged; it must not be reported
        # as cut short just because both conditions stopped being true at once.
        first = _make_sim(interval=100, NrTS=50000, EndCriteria=1e-6)
        stop = _run(first, 'tie_probe')['timesteps']
        self.assertEqual(first.GetTerminationReasonString(), 'converged')

        exact = _make_sim(interval=100, NrTS=stop, EndCriteria=1e-6)
        st = _run(exact, 'tie_exact')
        self.assertEqual(st['timesteps'], stop)
        self.assertEqual(exact.GetTerminationReasonString(), 'converged',
                         'NrTS==%d, the timestep the run converges at, was '
                         'reported as %r' % (stop, exact.GetTerminationReasonString()))
        self.assertEqual(exact.GetTerminationReason(), 1)
        self.assertEqual(st['reason_code'], 1)

    def test_a_run_that_missed_the_criteria_is_never_called_converged(self):
        # the mirror image of the test above: an end criteria that cannot be
        # met has to leave every reported reason as "not converged", whichever
        # limit stopped the run
        for kw, tag in (({'NrTS': 500}, 'nc_ts'),
                        ({'NrTS': 10 ** 8, 'MaxRunTime': 2}, 'nc_rt')):
            with self.subTest(**kw):
                fdtd = _make_sim(interval=100, EndCriteria=1e-30, **kw)
                st = _run(fdtd, tag)
                self.assertNotEqual(fdtd.GetTerminationReasonString(), 'converged')
                self.assertNotEqual(st['reason_code'], 1)


class Test_EndCriteriaRange(unittest.TestCase):
    """An end criteria of 1 or more is met before anything has been simulated."""

    def test_an_end_criteria_of_one_or_more_cannot_report_a_run_of_nothing(self):
        # change starts at 1 and the loop runs while change>endCriteria, so an
        # end criteria of >=1 stops the loop at timestep 0 -- and the run then
        # reports itself "converged" having simulated nothing at all, which is
        # the same mislabel this feature exists to prevent. The setter refuses
        # the value, so the run is the one the previous (valid) criteria asks for.
        ref = _make_sim(interval=100, NrTS=2000, EndCriteria=1e-6)
        ref_ts = _run(ref, 'endcrit_ref')['timesteps']
        self.assertGreater(ref_ts, 0)

        for bad in (1.0, 2.5):
            with self.subTest(EndCriteria=bad):
                fdtd = _make_sim(interval=100, NrTS=2000, EndCriteria=1e-6)
                fdtd.SetEndCriteria(bad)
                st = _run(fdtd, 'endcrit_%g' % bad)
                self.assertGreater(st['timesteps'], 0,
                                   'EndCriteria=%g simulated nothing and reported %r'
                                   % (bad, fdtd.GetTerminationReasonString()))
                self.assertEqual(st['timesteps'], ref_ts)


class Test_EndCriteriaCadence(unittest.TestCase):
    """The end criteria is sampled on a timestep cadence, not a wall-clock one."""

    # (name, kwargs for _make_sim, intervals). The loop advances by up to one
    # Nyquist interval of timesteps at a time, so the interesting cases are the
    # check intervals SHORTER than that: an unclamped chunk then overshoots the
    # first evaluation and offsets every later one by the remainder. The 1 GHz
    # bed advances 43 timesteps per iteration, the 6 GHz bed 7.
    BEDS = (('6GHz_21', {'n': 21, 'half': 10.0, 'f0': 6e9, 'fc': 3e9}, (3, 5, 37, 100)),
            ('1GHz_41', {'n': 41, 'half': 20.0, 'f0': 1e9, 'fc': 500e6}, (37, 100)))

    def test_converged_run_stops_on_the_cadence(self):
        # A converged run can only stop at a timestep where the criteria was
        # evaluated, i.e. at a multiple of the interval. Under a wall-clock
        # cadence the stopping timestep is wherever the machine happened to be
        # when the timer expired, which lands on a multiple of the interval
        # only by coincidence.
        for name, bed, intervals in self.BEDS:
            for interval in intervals:
                with self.subTest(bed=name, interval=interval):
                    fdtd = _make_sim(interval=interval, NrTS=200000,
                                     EndCriteria=1e-6, **bed)
                    st = _run(fdtd, 'cadence_%s_%d' % (name, interval))
                    self.assertEqual(fdtd.GetTerminationReasonString(), 'converged')
                    self.assertEqual(st['timesteps'] % interval, 0,
                                     'stopped at timestep %d, not a multiple of %d'
                                     % (st['timesteps'], interval))

    def test_repeated_runs_stop_at_the_same_timestep(self):
        runs = [_run(_make_sim(interval=64, NrTS=50000, EndCriteria=1e-6), 'rep%d' % i)
                for i in range(3)]
        self.assertEqual(len({r['timesteps'] for r in runs}), 1)

    def test_thread_count_does_not_change_the_stopping_timestep(self):
        # The invariant itself, on a bed small enough to run unconditionally.
        # It cannot skip: a machine on which extra threads buy nothing still
        # has to give the same answer, and a machine on which they do is where
        # a wall-clock cadence would show. The measured rates are reported so a
        # reader can tell which of the two this machine was.
        runs = [_run(_make_sim(interval=100, NrTS=50000, EndCriteria=1e-6),
                     'nthr%d' % t, numThreads=t) for t in (1, 4)]
        rates = ['%.0f TS/s' % (r['timesteps'] / r['wall']) for r in runs]
        for r in runs:
            self.assertEqual(r['reason_code'], 1)
        self.assertEqual(runs[0]['timesteps'], runs[1]['timesteps'],
                         'the stopping timestep moved with the thread count: '
                         '%d at 1 thread, %d at 4 threads (rates %s)'
                         % (runs[0]['timesteps'], runs[1]['timesteps'], rates))

    @unittest.skipUnless(SLOW, 'set OPENEMS_SLOW_TESTS=1 to run (two simulations of ~15 s each)')
    def test_stopping_timestep_does_not_depend_on_machine_speed(self):
        # The same invariant on a bed big enough that each run lasts longer
        # than the 4 s window the end criteria used to be sampled on, which is
        # what makes it a regression test for the defect and not only for the
        # invariant. Nothing here is allowed to skip: whatever this machine
        # does with the thread count, the answer has to come out the same.
        runs = [_run(_make_sim(interval=100, n=141, half=70.0, NrTS=200000,
                               EndCriteria=1e-9), 'threads%d' % t, numThreads=t)
                for t in (1, 8)]
        detail = ', '.join('%d threads: %d TS in %.1f s'
                           % (t, r['timesteps'], r['wall'])
                           for t, r in zip((1, 8), runs))
        for r in runs:
            self.assertEqual(r['reason_code'], 1,
                             'a run was cut off by NrTS instead of converging (%s)' % detail)
        self.assertEqual(runs[0]['timesteps'], runs[1]['timesteps'],
                         'the stopping timestep moved with the thread count (%s)' % detail)


if __name__ == '__main__':
    unittest.main()
