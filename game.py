import pygame
from pygame.locals import *

from snake import Snake
from point import Point
from config import *

import sys


class Game:
    def __init__(self, width, height, fps):
        self._running = True
        self._display_surf = None
        self._clock = None
        self.fps = fps
        self.size = self.width, self.height = width, height

        self.snake = None

        self.point_group = pygame.sprite.Group()
        self.point = None

        self.points = 0

    def init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._clock = pygame.time.Clock()
        self._running = True

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

        self.snake.on_event(event)

    def update(self):
        self.snake.update()
        self.point_group.update()
        self._score()

    def render(self):
        self.snake.draw(self._display_surf)
        self.point_group.draw(self._display_surf)

    def cleanup(self):
        pygame.quit()

    def execute(self):
        if self.init() == False:
            self._running = False

        self.snake = Snake(SNAKE_BODY_COLOR,
                           SNAKE_WIDTH,
                           SNAKE_HEIGHT,
                           pygame.Vector2(10, 10))

        self.point = Point(POINT_COLOR, POINT_WIDTH, POINT_HEIGHT)
        self.point_group.add(self.point)
        self.point.new_position()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.update()
            self._display_surf.fill((0, 0, 0))
            self.render()
            pygame.display.flip()
            self._clock.tick(self.fps)
        self.cleanup()

    def _score(self):
        if pygame.sprite.collide_rect(self.snake.head, self.point):
            self.points += 1
            self.point.new_position()
            self.snake.add_body()


if __name__ == '__main__':
    game = Game(SCREEN_SIZE[0], SCREEN_SIZE[1], FPS)
    game.execute()


# If the sprite is the last one in the group, it should remove the coordinate from the tuple of coordinates
# Tuple of coordinates should contain:
# 1. The point where sprite should update its movement
# 2. The new direction of the movement
# The game should be updated and then there should be done the check
# If the last sprite in the group is at the position of the last coordinate then, this coordinates should be removed
