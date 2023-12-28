from typing import Optional, List, Tuple

import os
import gc
import sys
import json
import tqdm
import time
import copy
import random
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def load_file_drawings(
    strokes_dir: str,
    file_name: str,
    index_lookup,
    all_drawings_strokes,
    progress: Optional[tqdm.tqdm] = None
): 
    print(f"starting {file_name}")
    file_path = os.path.join(strokes_dir, file_name)
    drawings = load_drawings(file_path)

    drawings_strokes = [
        drawing_to_strokes(drawing)
        for drawing in drawings
    ]

    num_prev_drawings = len(all_drawings_strokes)
    indices = [
        [num_prev_drawings + drawing_index, stroke_index]
        for drawing_index in range(len(drawings_strokes))
        for stroke_index in range(len(drawings_strokes[drawing_index]))
    ]

    index_lookup.extend(indices)
    all_drawings_strokes.extend(drawings)

    print(f"finished {file_name}")
    print(f"index_lookup: {sys.getsizeof(index_lookup)}")
    print(f"all_drawings_strokes: {sys.getsizeof(all_drawings_strokes)}")
    print(f"traced_memory: {tracemalloc.get_traced_memory()}")

    if progress is not None:
        progress.update(1)


def load_drawings_strokes(
    strokes_dir: str
) -> Tuple[
        List[List[List[float]]],
        List[List[int]]
    ]:
    """
    drawings_strokes[drawing_index][stroke_index] = (x, y, pen_down)
    index_to_drawing_stroke_indices[index] = (drawing_index, stroke_index)

    :param strokes_dir: _description_
    :return: _description_
    """
    tracemalloc.start()
    all_drawings_strokes = []
    index_lookup = []

    #with ThreadPoolExecutor(max_workers=3) as executor:
    if True:
        file_names = os.listdir(strokes_dir)
        progress = tqdm.tqdm(desc="Classes loaded", total=len(file_names))
        for file_name in file_names:
            #executor.submit(load_file_drawings, strokes_dir, file_name, index_lookup, all_drawings_strokes, progress)
            load_file_drawings(strokes_dir, file_name, index_lookup, all_drawings_strokes)#, progress)

    return all_drawings_strokes, index_lookup


def split_drawings_strokes(
    index_lookup: List[Tuple[int, int]],
    test_size: float,
    shuffle: bool = True
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    indices = list(range(len(index_lookup)))
    if shuffle:
        random.shuffle(indices)

    split_index = int(test_size * len(index_lookup))
    test_indices = indices[:split_index]
    train_indices = indices[split_index:]

    test_index_lookup = [index_lookup[index] for index in test_indices]
    train_index_lookup = [index_lookup[index] for index in train_indices]

    return train_index_lookup, test_index_lookup


def load_drawings(file_path: str) -> List[List[float]]:
    drawings = []

    with open(file_path, "r") as file:
        lines = file.readlines()  # reading lines eagerly doesn't take that long
        for line in lines:
            data = json.loads(line)
            if data["recognized"]:
                drawings.append(data["drawing"])

    return drawings


def drawing_to_strokes(drawing: List[List[int]], drawings = None):
    """
    for stroke in drawing:
        yield [
            [x, y, 0.0]
            for x, y in zip(*stroke)
        ] + [
            [0.0, 0.0, 1.0]
        ]
    """
    strokes = [
        [
            [x, y, 0.0]
            for x, y in zip(*stroke)
        ] + [
            [0.0, 0.0, 1.0]
        ]
        for stroke in drawing
    ]

    if drawings is not None:
        drawings.append(strokes)
    else:
        return strokes
