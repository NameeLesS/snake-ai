import pygame
from pygame.locals import *

from snake import Snake
from point import Point
from score import Score
from config import *


class Game:
    def __init__(self, size, fps):
        self._running = True
        self._display_surf = None
        self._clock = None
        self.fps = fps
        self.size = size

        self.snake = None

        self.point_group = pygame.sprite.Group()
        self.point = None

        self.score = None

    def init(self):
        pygame.init()
        pygame.font.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._clock = pygame.time.Clock()
        self._running = True

        self.snake = Snake(SNAKE_BODY_COLOR,
                           SNAKE_WIDTH,
                           SNAKE_HEIGHT,
                           pygame.Vector2(10, 10))
        self.point = Point(POINT_COLOR, POINT_WIDTH, POINT_HEIGHT)
        self.point_group.add(self.point)
        self.score = Score(30)

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
        self.score.draw(self._display_surf)

    def cleanup(self):
        pygame.quit()

    def execute(self):
        if self.init() == False:
            self._running = False

        self.point.new_position()

        while self._running:
            self.on_event(pygame.event.poll())
            self.update()
            self._display_surf.fill(BACKGROUND_COLOR)
            self.render()
            pygame.display.flip()
            self._clock.tick(self.fps)
        self.cleanup()

    def _score(self):
        if pygame.sprite.collide_rect(self.snake.head, self.point):
            self.score.add_score(1)
            self.point.new_position()
            self.snake.extend_body()


if __name__ == '__main__':
    game = Game(SCREEN_SIZE, FPS)
    game.execute()
