from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point

def build_world(dt = 0.1):
    # dt: time steps in terms of seconds. In other words, 1/dt is the FPS.
    
    width = 120
    height = 120
    ppm = 6
    w = World(dt, width, height, ppm)  # The world is width x height. ppm is the pixels per meter.

    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
    # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
    # For both of these objects, we give the center point and the size.
    w.add(Painting(Point(113.5, 93.5), Point(97, 57), 'gray80')) # We build a sidewalk.
    w.add(RectangleBuilding(Point(114.5, 94.5), Point(95, 55))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

    # Let's repeat this for 4 different RectangleBuildings.
    w.add(Painting(Point(12.5, 93.5), Point(85, 57), 'gray80'))
    w.add(RectangleBuilding(Point(11.5, 94.5), Point(83, 55)))

    w.add(Painting(Point(12.5, 21.5), Point(85, 67), 'gray80'))
    w.add(RectangleBuilding(Point(11.5, 20.5), Point(83, 65)))

    w.add(Painting(Point(113.5, 21.5), Point(97, 67), 'gray80'))
    w.add(RectangleBuilding(Point(114.5, 20.5), Point(95, 65)))

    # Let's also add some crossings, because why not.
    for i in range(2, 120, 4):
        w.add(Painting(Point(i, 60), Point(2, 0.5), 'white'))
        w.add(Painting(Point(60, i), Point(0.5, 2), 'white'))
    w.add(Painting(Point(60, 60), Point(10, 10), 'gray'))

    return w