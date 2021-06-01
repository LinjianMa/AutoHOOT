# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Dense(object):
    pass


class Compressed(object):
    pass


class DenseFormat(object):
    def __eq__(self, other):
        if isinstance(other, DenseFormat):
            return True
        else:
            return False


class SparseFormat(object):
    """ Sparse tensor data format.
    Parameters
    ----------
    mode_formats: a list containing the format of each mode ("dense" or "compressed")
    mode_order: (optional) a list specifying the order in which modes should be stored
    """
    def __init__(self, mode_formats, mode_order=None):
        for mode in mode_formats:
            assert isinstance(mode, (Dense, Compressed))
        self.mode_formats = mode_formats
        if mode_order is None:
            self.mode_order = list(range(len(mode_formats)))
        else:
            self.mode_order = mode_order

    def __eq__(self, other):
        if isinstance(other, DenseFormat):
            return False
        if self.mode_order != other.mode_order:
            return False
        return all(
            type(mode1) == type(mode2)
            for mode1, mode2 in zip(self.mode_formats, other.mode_formats))


dense = Dense()
compressed = Compressed()
