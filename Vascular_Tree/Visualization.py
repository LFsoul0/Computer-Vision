from typing import Tuple
import numpy as np
from mayavi import mlab

def Draw3DContours(data : np.ndarray, 
                   color : Tuple[float, float, float] = (1.0, 1.0, 1.0), 
                   line_width : float = 1.0,
                   opacity : float = 0.1) :
    data = data.astype(np.uint8)
    data[data != 0] = 255

    mlab.contour3d(data, 
                  color = color, 
                  contours = [255],
                  line_width = line_width,
                  opacity = opacity)

def Show() :
    mlab.show()
