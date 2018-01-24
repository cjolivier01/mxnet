# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""Profiler setting methods."""
from __future__ import absolute_import
import ctypes
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array

def profiler_set_config(flags):
    """Set up the configure of profiler.

    Parameters
    ----------
    flags : list of key/value pair tuples
        Indicates configuration parameters
          profile_all : boolean, all profile types enabled
          profile_symbolic : boolean, whether to profile symbolic operators
          profile_imperative : boolean, whether to profile imperative operators
          profile_memory : boolean, whether to profile memory usage
          profile_api : boolean, whether to profile the C API
          file_name : string, output file for profile data
          continuous_dump : boolean, whether to periodically dump profiling data to file
          dump_period : float, seconds between profile data dumps
    """
    check_call(_LIB.MXSetProfilerConfig(len(flags),
                                        c_str_array([key for key, _ in flags]),
                                        c_str_array([str(val) for _, val in flags])))


def profiler_set_state(state='stop'):
    """Set up the profiler state to 'run' or 'stop'.

    Parameters
    ----------
    state : string, optional
        Indicates whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    """
    state2int = {'stop': 0, 'run': 1}
    check_call(_LIB.MXSetProfilerState(ctypes.c_int(state2int[state])))

def dump_profile():
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally.
    """
    check_call(_LIB.MXDumpProfile())

def profiler_pause():
    check_call(_LIB.MXProfilePause(int(1)))

def profiler_resume():
    check_call(_LIB.MXProfilePause(int(0)))


class Domain(object):
    """Profiling domain, used to group sub-objects like tasks, counters, etc into categories
    Serves as part of 'categories' for chrome://tracing
    Note: Domain handles are never destroyed.
    """
    def __init__(self, name):
        """Profiling Domain class constructor
            Parameters
            ----------
            name : string
                Name of the domain
        """
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateDomain(c_str(self.name), ctypes.byref(self.handle)))

    def __str__(self):
        return self.name

    def new_task(self, name):
        """Create new Task object owned by this domain
            Parameters
            ----------
            name : string
                Name of the task
        """
        return Task(self, name)

    def new_frame(self, name):
        """Create new Frame object owned by this domain
            Parameters
            ----------
            name : string
                Name of the frame
        """
        return Frame(self, name)

    def new_counter(self, name, value=None):
        """Create new Counter object owned by this domain
            Parameters
            ----------
            name : string
                Name of the counter
        """
        return Counter(self, name, value)

    def new_marker(self, name):
        """Create new Marker object owned by this domain
            Parameters
            ----------
            name : string
                Name of the marker
        """
        return Marker(self, name)

class Task(object):
    """Profiling Task class
    A task is a logical unit of work performed by a particular thread.
    Tasks can nest; thus, tasks typically correspond to functions, scopes, or a case block
    in a switch statement.
    You can use the Task API to assign tasks to threads.
    """
    def __init__(self, domain, name):
        """Profiling Task class constructor.
            Parameters
            ----------
            domain : Domain object
                Domain to which this object belongs
            name : string
                Name of the task
        """
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateTask(domain.handle,
                                            c_str(self.name),
                                            ctypes.byref(self.handle)))

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))

    def start(self):
        """Start timing scope for this object"""
        check_call(_LIB.MXProfileDurationStart(self.handle))

    def stop(self):
        """Stop timing scope for this object"""
        check_call(_LIB.MXProfileDurationStop(self.handle))

    def __str__(self):
        return self.name


class Frame(object):
    """Profiling Frame class
    Use the frame API to insert calls to the desired places in your code and analyze
    performance per frame, where frame is the time period between frame begin and end points.
    When frames are displayed in Intel VTune Amplifier, they are displayed in a
    separate track, so they provide a way to visually separate this data from normal task data.
    """
    def __init__(self, domain, name):
        """Profiling Frame class constructor
            Parameters
            ----------
            domain : Domain object
                Domain to which this object belongs
            name : string
                Name of the frame
        """
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateFrame(domain.handle,
                                             c_str(self.name),
                                             ctypes.byref(self.handle)))

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))

    def start(self):
        """Start timing scope for this object"""
        check_call(_LIB.MXProfileDurationStart(self.handle))

    def stop(self):
        """Stop timing scope for this object"""
        check_call(_LIB.MXProfileDurationStop(self.handle))

    def __str__(self):
        return self.name


class Event(object):
    """Profiling Event class
    The event API is used to observe when demarcated events occur in your application, or to
    identify how long it takes to execute demarcated regions of code. Set annotations in the
    application to demarcate areas where events of interest occur.
    After running analysis, you can see the events marked in the Timeline pane.
    Event API is a per-thread function that works in resumed state.
    This function does not work in paused state.
    """
    def __init__(self, name):
        """Profiling Event class constructor
            Parameters
            ----------
            name : string
                Name of the event
        """
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateEvent(c_str(self.name), ctypes.byref(self.handle)))

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))

    def start(self):
        """Start timing scope for this object"""
        check_call(_LIB.MXProfileDurationStart(self.handle))

    def stop(self):
        """Stop timing scope for this object"""
        check_call(_LIB.MXProfileDurationStop(self.handle))

    def __str__(self):
        return self.name


class Counter(object):
    """Profiling Counter class
    The counter event can track a value as it changes over time.
    """
    def __init__(self, domain, name, value=None):
        """Profiling Counter class constructor.
        The counter event can track a value as it changes over time.
            Parameters
            ----------
            domain : Domain object
                Domain to which this object belongs
            name : string
                Name of the counter
            value: integer, optional
                Initial value of the counter
        """
        self.name = name
        self.handle = ProfileHandle()
        check_call(_LIB.MXProfileCreateCounter(domain.handle,
                                               c_str(name),
                                               ctypes.byref(self.handle)))
        if value is not None:
            self.set_value(value)

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXProfileDestroyHandle(self.handle))


    def set_value(self, value):
        """Set counter value.
            Parameters
            ----------
            value : int
                Value for the counter
        """
        check_call(_LIB.MXProfileSetCounter(self.handle, int(value)))

    def increment(self, value_change):
        """Increment counter value.
            Parameters
            ----------
            value_change : int
                Amount by which to add to the counter
        """
        check_call(_LIB.MXProfileAdjustCounter(self.handle, int(value_change)))

    def decrement(self, value_change):
        """Decrement counter value.
            Parameters
            ----------
            value_change : int
                Amount by which to subtract from the counter
        """
        check_call(_LIB.MXProfileAdjustCounter(self.handle, -int(value_change)))

    def __iadd__(self, value_change):
        self.increment(value_change)
        return self

    def __isub__(self, value_change):
        self.decrement(value_change)
        return self

    def __str__(self):
        return self.name


class Marker(object):
    """Set marker for an instant in time"""
    def __init__(self, domain, name):
        """Profiling Marker class constructor
        The marker event marks a particular instant in time across some scope boundaries.
            Parameters
            ----------
            domain : Domain object
                Domain to which this object belongs
            name : string
                Name of the marker
        """
        self.name = name
        self.domain = domain

    def mark(self, scope='process'):
        """Set up the profiler state to record operator.

        Parameters
        ----------
        scope : string, optional
            Indicates what scope the marker should refer to.
            Can be 'global', 'process', thread', task', and 'marker'
            Default is `process`.
        """
        check_call(_LIB.MXProfileSetMarker(self.domain.handle, c_str(self.name), c_str(scope)))
