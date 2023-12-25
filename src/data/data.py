from typing import Optional, List, Tuple

import os
import json
import random


def load_drawings_strokes(
    strokes_dir: str
) -> Tuple[
        List[List[Tuple[float, float, float]]],
        List[Tuple[int, int]]
    ]:
    """
    drawings_strokes[drawing_index][stroke_index] = (x, y, pen_down)
    index_to_drawing_stroke_indices[index] = (drawing_index, stroke_index)

    :param strokes_dir: _description_
    :return: _description_
    """
    all_drawings_strokes = []
    index_to_drawing_stroke_indices = []
    for file_name in os.listdir(strokes_dir):
        file_path = os.path.join(strokes_dir, file_name)
        drawings = load_drawings(file_path)

        drawings_strokes = [
            drawing_to_strokes(drawing)
            for drawing in drawings
        ]

        num_prev_drawings = len(all_drawings_strokes)
        index_to_drawing_stroke_indices.extend([
            (num_prev_drawings + drawing_index, stroke_index)
            for drawing_index in range(len(drawings_strokes))
            for stroke_index in range(len(drawings_strokes[drawing_index]))
        ])
        all_drawings_strokes.extend(drawings_strokes)

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
            data["recognized"]

            drawings.append(data["drawing"])

    return drawings


def drawing_to_strokes(drawing: List[List[int]]):
    return [
        [
            (x, y, 0.0)
            for x, y in zip(*stroke)
        ] + [
            (0.0, 0.0, 1.0)
        ]
        for stroke in drawing
    ]
