from typing import Optional, List, Tuple

import os
import json
import tqdm
import random
from concurrent.futures import ThreadPoolExecutor

def delete_lists(list1):
    if list1 is None:
        return
    for item in list1:
        if isinstance(item, list):
            delete_lists(item)
    del list1


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
    all_drawings_strokes = []
    index_to_drawing_stroke_indices = []
    def load_file_drawings(
        file_name: str,
        ds,
        idsi,
        progress: Optional[tqdm.tqdm] = None
    ):
        file_path = os.path.join(strokes_dir, file_name)
        drawings = load_drawings(file_path)

        drawings_strokes = [
            drawing_to_strokes(drawing)
            for drawing in drawings
        ]
        del drawings

        num_prev_drawings = len(all_drawings_strokes)
        indices = [
            [num_prev_drawings + drawing_index, stroke_index]
            for drawing_index in range(len(drawings_strokes))
            for stroke_index in range(len(drawings_strokes[drawing_index]))
        ]
        idsi.extend(indices)
        delete_lists(indices)
        ds.extend(drawings_strokes)
        delete_lists(drawings_strokes)

        if progress is not None:
            progress.update(1)

    with ThreadPoolExecutor(max_workers=None) as executor:
        file_names = os.listdir(strokes_dir)
        progress = tqdm.tqdm(desc="Classes loaded", total=len(file_names))
        for file_name in file_names:
            executor.submit(load_file_drawings, file_name, all_drawings_strokes, index_to_drawing_stroke_indices, progress)

    return all_drawings_strokes, index_to_drawing_stroke_indices


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
        for line in file:
            data = json.loads(line)
            if data["recognized"]:
                drawings.append(data["drawing"])

    return drawings


def drawing_to_strokes(drawing: List[List[int]]):
    return [
        [
            [x, y, 0.0]
            for x, y in zip(*stroke)
        ] + [
            [0.0, 0.0, 1.0]
        ]
        for stroke in drawing
    ]
