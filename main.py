#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  :   mrx
@Contact :   mengruxing@gmail.com
@Date    :   2020/11/20
@Desc    :   
"""

from __future__ import division, absolute_import, print_function, unicode_literals

import cv2
import math
import numba as nb
import numpy as np
import time

from threading import Lock, Condition, Thread
from multiprocessing import Process, Queue


class FPS(object):

    def __init__(self, length=256):
        self.length = length
        self.time_list = [time.time()]

    def __str__(self):
        return 'FPS: {} len: {}'.format(self.fps, self.length)

    def __call__(self, *args, **kwargs):
        self.update()
        return self.fps

    @property
    def fps(self):
        return self.length / (self.time_list[-1] - self.time_list[0])

    def update(self):
        self.time_list.append(time.time())
        self.time_list = self.time_list[-self.length:]


class EmptyCondition(object):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def wait(self):
        pass

    def notify(self):
        pass


class Reader(Thread):

    def __init__(self, src=None, notice=None, wait=False, mutex=None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture()
        self.cache = False, None
        self.keep_running = True
        self.notice = notice if notice is not None else Condition(mutex if mutex is not None else Lock()) if wait else EmptyCondition()
        self.read = self.read if wait else lambda: self.cache
        self.open(src=src)

    def __del__(self):
        self.release()

    def __call__(self, *args, **kwargs):
        return self.read()

    def release(self):
        self.keep_running = False
        return self.cap.release()

    def read(self):
        with self.notice:
            self.notice.wait()
        return self.cache

    def get(self, prop_id):
        return self.cap.get(propId=prop_id)

    def set(self, prop_id, value):
        return self.cap.set(propId=prop_id, value=value)

    def open(self, src):
        if src is not None and self.cap.open(src):
            self.cache = self.cap.read()
            self.start()
            return True
        return False

    def run(self) -> None:
        while self.keep_running:
            self.cache = self.cap.read()
            with self.notice:
                self.notice.notify()


class CameraReader(Process):

    def __init__(self, queue, camera_list=(), name=''):
        super().__init__(name='CameraReader<{}>'.format(name), daemon=True)
        self.queue = queue
        self.camera_list = camera_list
        self.keep_running = True

    def run(self) -> None:
        notice = Condition(Lock())
        readers = [Reader(src=src, notice=notice) for src in self.camera_list]
        while self.keep_running:
            with notice:
                notice.wait()
            self.queue.put([reader.read() for reader in readers])


@nb.jit
def recycle_motion(cap_size, little_steps=30, big_steps=30, cycle=30, max_scale=2., min_scale=1.):
    scope = (max_scale - min_scale) / 2.
    offset = scope + min_scale
    while True:
        for cap_id in range(cap_size):
            for t in range(little_steps):
                yield cap_id, offset - scope
            for t in range(cycle + 1):
                yield cap_id, offset - scope * math.cos(math.pi * t / cycle)
            for t in range(big_steps):
                yield cap_id, offset + scope
            for t in range(cycle + 1):
                yield cap_id, offset + scope * math.cos(math.pi * t / cycle)


class StreamPreviewer(object):

    def __init__(self, camera_list=(), size=(640, 480), scale=2):
        self.camera_list = camera_list
        self.width, self.height = self.size = size
        self.len = len(camera_list)
        self.column_num = int(math.ceil(math.sqrt(self.len)))
        self.row_num = self.column_num if self.column_num * (self.column_num - 1) < self.len else self.column_num - 1
        self.motion = recycle_motion(self.len, max_scale=min(self.row_num, scale))
        self.empty = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        self.animation = self.animation if self.row_num > 1 else lambda framework, frames: None

    def animation(self, framework, frames):
        f_id, scale = next(self.motion)
        row_id, column_id = divmod(f_id, self.column_num)
        height, width = int(self.height * scale), int(self.width * scale)
        point_h = int(self.height * row_id * (1 - (scale - 1) / (self.row_num - 1)))
        point_w = int(self.width * column_id * (1 - (scale - 1) / (self.column_num - 1)))
        framework[point_h:point_h + height, point_w:point_w + width] = cv2.resize(frames[f_id], (width, height))

    def hstack(self, frames):
        frames += [self.empty] * (self.column_num - len(frames))
        return np.hstack(frames)

    def gen_frames(self):
        notice = Condition(Lock())
        readers = [Reader(v, notice=notice) for v in self.camera_list]
        while True:
            with notice:
                notice.wait()
            origin_frames, resize_frames = [], []
            for ret, frame in [reader.read() for reader in readers]:
                origin_frames.append(frame if ret else self.empty)
                resize_frames.append(cv2.resize(frame, self.size) if ret else self.empty)
            yield origin_frames, resize_frames

    def gen_frames_with_process(self):
        queue = Queue(maxsize=8)
        camera_reader = CameraReader(queue=queue, camera_list=self.camera_list)
        camera_reader.start()
        while True:
            origin_frames, resize_frames = [], []
            for ret, frame in queue.get():
                origin_frames.append(frame if ret else self.empty)
                resize_frames.append(cv2.resize(frame, self.size) if ret else self.empty)
            yield origin_frames, resize_frames

    def run(self):
        fps = FPS(100)
        gen_frames = self.gen_frames()
        while True:
            origin_frames, resize_frames = next(gen_frames)
            framework = np.vstack(
                [self.hstack(resize_frames[i:i + self.column_num]) for i in range(0, self.len, self.column_num)])
            self.animation(framework, origin_frames)  # animation
            cv2.putText(framework, 'FPS: {:.2f}'.format(fps()), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                        thickness=2)
            cv2.imshow('preview', framework)
            if cv2.waitKey(1) == 27:
                break


def load_cameras(file_path='cameras.txt'):

    def trans(s):
        try:
            return int(s)
        except ValueError:
            return s

    with open(file_path) as f:
        return [trans(line.strip()) for line in f.readlines()]


if __name__ == '__main__':
    StreamPreviewer(camera_list=load_cameras(file_path='cameras.txt')).run()
