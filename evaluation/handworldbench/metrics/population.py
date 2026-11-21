"""Reference geometry determines which hand instances enter image metrics."""
import math


def reference_hand_sides(frames):
    sides=set()
    for frame in frames:
        for hand in frame.get('hands',[]):
            for point,valid in zip(hand['joints_2d'],hand['valid_2d']):
                x,y=point
                if valid and math.isfinite(x) and math.isfinite(y) and 0<=x<frame['width'] and 0<=y<frame['height']:
                    sides.add(hand['side'])
                    break
    return sorted(sides)
