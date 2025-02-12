# MIT License
#
# Copyright (c) 2024-2025 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Literal, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as _np
import pandas as _pd
from pynwb import (
    NWBHDF5IO as _NWBHDFIO,
    NWBFile as _NWBFile,
)


PathLike = str | Path


@dataclass(frozen=True)
class NWBMetadata:
    session_id: str
    session_description: str
    session_notes: str
    subject_id: str
    subject_DoB: str
    subject_age: str
    subject_sex: str


@dataclass(frozen=True)
class NWBData:
    downsampled: bool
    metadata: NWBMetadata
    trials: Optional[_pd.DataFrame]
    daq: _pd.DataFrame
    imaging: _pd.DataFrame
    body_video: Optional[_pd.DataFrame]
    face_video: Optional[_pd.DataFrame]
    eye_video: Optional[_pd.DataFrame]
    pupil: Optional[_pd.DataFrame]


def read_metadata(
    nwbfile: _NWBFile,
) -> NWBMetadata:
    # TODO: better including trials/acquisition/ROIs/keypoints metadata
    metadata = {
        'session_id': nwbfile.session_id,
        'session_description': nwbfile.session_description,
        'session_notes': nwbfile.notes,
        'subject_id': nwbfile.subject.subject_id,
        'subject_DoB': nwbfile.subject.date_of_birth.strftime('%Y-%m-%d'),
        'subject_age': nwbfile.subject.age,
        'subject_sex': nwbfile.subject.sex
    }
    return NWBMetadata(**metadata)


def read_trials(
    nwbfile: _NWBFile,
    downsampled: bool = True
) -> Optional[_pd.DataFrame]:
    if downsampled:
        if 'trials' not in nwbfile.get_processing_module('downsampled').data_interfaces:
            return None
        entry = nwbfile.get_processing_module('downsampled').get_data_interface('trials')
    else:
        entry = nwbfile.trials
        if entry is None:
            return None
    trials = entry.to_dataframe()
    trials.index.name = None
    return trials


def read_acquisition(
    nwbfile: _NWBFile,
    downsampled: bool = True
) -> _pd.DataFrame:
    if downsampled:
        root = nwbfile.get_processing_module('downsampled')
        node_names = tuple(root.data_interfaces)

        def _get(node):
            return root.get_data_interface(node)
    else:
        node_names = tuple(nwbfile.acquisition.keys())

        def _get(node):
            return nwbfile.get_acquisition(node)

    data = dict()
    time = None
    for node in node_names:
        # exclude some of the nodes
        if node.startswith('widefield_') or node.endswith('_video') or node.endswith('_keypoints'):
            continue
        elif node in ('eye_position', 'pupil_tracking', 'trials'):
            continue
        entry = _get(node)
        values = _np.array(entry.data)
        if time is None:
            time = _pd.Index(data=_np.array(entry.timestamps), name='time')
        if node.startswith('state_'):
            values = values.astype(_np.int16)
        data[node] = values
    return _pd.DataFrame(data=data, index=time)


def read_video_tracking(
    nwbfile: _NWBFile,
    view: Literal['body', 'face', 'eye', 'pupil'] = 'body',
    downsampled: bool = True
) -> Optional[_pd.DataFrame]:
    if view in ('body', 'face', 'eye'):
        parent_name = f"{view}_video_keypoints"
        if downsampled:
            root = nwbfile.get_processing_module('downsampled')
        else:
            root = nwbfile.get_processing_module('behavior')
        if parent_name not in root.data_interfaces:
            return None
        entries = root.get_data_interface(parent_name)

        data = dict()
        time = None
        for kpt in entries.pose_estimation_series.keys():
            entry = entries.get_pose_estimation_series(kpt)
            if time is None:
                time = _pd.Index(data=_np.array(entry.timestamps), name='time')
            values = _np.array(entry.data)
            data[kpt, 'x'] = values[:, 0]
            data[kpt, 'y'] = values[:, 1]
            if not downsampled:
                data[kpt, 'likelihood'] = _np.array(entries.get_pose_estimation_series(kpt).confidence)
    elif view == 'pupil':
        if downsampled:
            root = nwbfile.get_processing_module('downsampled')
        else:
            root = nwbfile.get_processing_module('behavior')
        if 'pupil_tracking' not in root.data_interfaces:
            return None

        data = dict()
        time = _pd.Index(
            data=_np.array(root.get_data_interface('pupil_tracking').get_timeseries('diameter').timestamps),
            name='time',
        )
        data['diameter'] = _np.array(root.get_data_interface('pupil_tracking').get_timeseries('diameter').data)
        data['center_x'] = _np.array(root.get_data_interface('eye_position').get_spatial_series('center_x').data)
        data['center_y'] = _np.array(root.get_data_interface('eye_position').get_spatial_series('center_y').data)
    else:
        raise ValueError(f"expected 'body', 'face', 'eye' or 'pupil', but got '{view}'")
    return _pd.DataFrame(data=data, index=time)


def read_roi_dFF(nwbfile: _NWBFile) -> _pd.DataFrame:
    dff_entry = nwbfile.get_processing_module('ophys').get_data_interface('DfOverF').get_roi_response_series('dFF')
    time = _np.array(dff_entry.timestamps)
    dff = _np.array(dff_entry.data)
    names = tuple(str(name) for name in _np.array(dff_entry.rois.table.get('roi_name')))
    # TODO: it may be needed at some time in the future
    # descs = tuple(str(desc) for desc in _np.array(dff_entry.rois.table.get('roi_description')))
    # roidescs = dict((name, desc) for name, desc in zip(names, descs))
    return _pd.DataFrame(data=dff, columns=names, index=time)


def load_from_file(
    nwbfilepath: PathLike,
    downsampled: bool = True
) -> NWBData:
    with _NWBHDFIO(nwbfilepath, mode='r') as src:
        nwbfile = src.read()
        data = dict()
        data['metadata'] = read_metadata(nwbfile)
        data['daq'] = read_acquisition(nwbfile, downsampled=downsampled)
        data['imaging'] = read_roi_dFF(nwbfile)
        data['trials'] = read_trials(nwbfile, downsampled=downsampled)
        for view in ('body', 'face', 'eye'):
            data[f'{view}_video'] = read_video_tracking(nwbfile, view=view, downsampled=downsampled)
        data['pupil'] = read_video_tracking(nwbfile, view='pupil', downsampled=downsampled)
    return NWBData(downsampled=downsampled, **data)
