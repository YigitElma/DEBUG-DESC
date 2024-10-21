import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from desc.examples import get
from desc.plotting import plot_surfaces
from manim import *
import numpy as np


class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation


class RandomShapeScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        eq = get("ARIES-CS")
        _, _, data = plot_surfaces(
            eq, rho=10, phi=np.pi / eq.NFP, theta=0, return_data=True
        )
        shapes = []
        circles = []
        animations = []
        for i in range(0, 10):
            points_x = data["rho_R_coords"][:, i].squeeze()
            points_y = -data["rho_Z_coords"][:, i].squeeze()

            # Combine x and y into Manim Points
            points = [np.array([x, y, 0]) for x, y in zip(points_x, points_y)]

            # Create the polygon with the points
            color = RED if i == 9 else BLUE

            random_shape = Polygon(*points, color=color)
            random_shape.move_to(ORIGIN).scale(2)
            circle = Circle(radius=0.2 * (i + 1), color=color)

            circles.append(circle)
            shapes.append(random_shape)
            animations.append(Transform(shapes[-1], circles[-1]))

        self.play(AnimationGroup(*animations, lag_ratio=0, run_time=3))
        self.wait(2)
        self.clear()

        shapes = []
        circles = []
        animations = []
        for i in range(0, 10):
            points_x = data["rho_R_coords"][:, i].squeeze()
            points_y = -data["rho_Z_coords"][:, i].squeeze()

            # Combine x and y into Manim Points
            points = [np.array([x, y, 0]) for x, y in zip(points_x, points_y)]

            # Create the polygon with the points
            color = RED if i == 9 else BLUE

            random_shape = Polygon(*points, color=color)
            random_shape.move_to(ORIGIN).scale(2)
            circle = Circle(radius=0.2 * (i + 1), color=color)

            circles.append(circle)
            shapes.append(random_shape)
            animations.append(Transform(circles[-1], shapes[-1]))

        self.play(AnimationGroup(*animations, lag_ratio=0, run_time=3))
        self.wait(2)
