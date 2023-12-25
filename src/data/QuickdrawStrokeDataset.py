from typing import Tuple, List

import torch
import numpy
import cairo

from competitive_drawing.train.utils.RandomResizePad import RandomResizePad


def strokes_to_raster(
    strokes: List[Tuple[float, float, float]],
    side: int = 50,
    line_diameter: int = 16,
    padding: int = 16
) -> numpy.ndarray:
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    if len(strokes) <= 0:
        return numpy.zeros((1, side, side), dtype=numpy.float32)

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

    # don't offset to center, not necessary (as of now)

    # clear background
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    for stroke in strokes:
        ctx.move_to(stroke[0][0], stroke[0][1])
        # skip first because of previous line
        # skip last because of stroke end data
        for i in range(1, len(stroke) - 1):
            ctx.line_to(stroke[i][0], stroke[i][1])
        ctx.stroke()

    data = surface.get_data()
    raster_image = numpy.copy(numpy.asarray(data, dtype=numpy.float32)[::4])
    raster_image = raster_image / 255.0
    raster_image = raster_image.reshape((side, side))
    raster_image = numpy.expand_dims(raster_image, axis=0)  # one channel image

    return raster_image


class QuickdrawStrokeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        drawings_strokes: List[List[Tuple[float, float, float]]],
        index_to_drawing_stroke_indices: List[Tuple[int, int]] = 50,
        image_size: int = 50
    ):
        self.drawings_strokes = drawings_strokes
        self.index_to_drawing_stroke_indices = index_to_drawing_stroke_indices
        self.image_size = image_size


    def __len__(self):
        return len(self.index_to_drawing_stroke_indices)


    """ TODO: save for when we train the RNN
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        drawing_index, stroke_index = self.index_to_drawing_stroke_indices[index]
    
        image_strokes = self.drawings_strokes[drawing_index][:stroke_index]
        image = strokes_to_raster(image_strokes, side=self.image_size)
        next_stroke = self.drawings_strokes[drawing_index][stroke_index]

        return torch.tensor(image), torch.tensor(next_stroke)
    """


    def __getitem__(self, index: int) -> torch.Tensor:
        drawing_index, stroke_index = self.index_to_drawing_stroke_indices[index]
    
        image_strokes = self.drawings_strokes[drawing_index][:stroke_index]
        image = strokes_to_raster(image_strokes, side=self.image_size)

        return torch.tensor(image, dtype=torch.float32)
    

if __name__ == "__main__":
    dataset = QuickdrawStrokeDataset("strokes")
    for image in dataset:
        pass
