from typing import Tuple, List

import os
import json
import torch
import numpy
import cairo
from torchvision import transforms

from competitive_drawing.train.utils.RandomResizePad import RandomResizePad


def strokes_to_raster(
    strokes: List[Tuple[float, float, float]],
    side: int = 50,
    line_diameter: int = 16,
    padding: int = 16,
    bg_color: Tuple[int, int, int] = (0,0,0),
    fg_color: Tuple[int, int, int] = (1,1,1)
):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    if len(strokes) <= 0:
        return numpy.zeros(side, dtype=numpy.float32)

    original_side = 256.0

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2.0 + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2.0, total_padding / 2.0)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    for stroke in strokes:
        ctx.move_to(stroke[0][0], stroke[0][1])
        for i in range(len(stroke)):  # may be double drawing first position
            ctx.line_to(stroke[i][0], stroke[i][1])
        ctx.stroke()

    data = surface.get_data()
    raster_image = numpy.copy(numpy.asarray(data)[::4])

    return raster_image


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


class QuickdrawStrokeDataset(torch.utils.data.Dataset):
    def __init__(self, strokes_dir: str, image_size: int = 50):
        (
            self.drawings_strokes,
            self.index_to_drawing_stroke_indices
        ) = load_drawings_strokes(strokes_dir)
        self.image_size = image_size


    def __len__(self):
        return len(self.index_to_drawing_stroke_indices)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        drawing_index, stroke_index = self.index_to_drawing_stroke_indices[index]
    
        image_strokes = self.drawings_strokes[drawing_index][:stroke_index]
        next_stroke = self.drawings_strokes[drawing_index][stroke_index]

        image = strokes_to_raster(image_strokes, side=self.image_size)

        return torch.tensor(image), torch.tensor(next_stroke)
    

if __name__ == "__main__":
    dataset = QuickdrawStrokeDataset("strokes")
    for image, next_stroke in dataset:
        print(next_stroke)
