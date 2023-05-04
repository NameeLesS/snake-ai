import pygame
import numpy as np
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
        self.terminate = False

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
        self.restart()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

        self.snake.on_event(event)

    def update(self):
        self.snake.update()
        self.point_group.update()
        self._score()
        self.terminate = self.snake.collides

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
            self.loop()
            if self.terminate:
                self.restart()
        self.cleanup()

    def loop(self):
        self.on_event(pygame.event.poll())
        self.update()
        self._display_surf.fill(BACKGROUND_COLOR)
        self.render()
        pygame.display.flip()
        self._clock.tick(self.fps)

    def restart(self):
        self.snake = Snake(SNAKE_BODY_COLOR,
                           SNAKE_WIDTH,
                           SNAKE_HEIGHT,
                           pygame.Vector2(10, 10))

        self.point_group.remove(self.point)
        self.point = Point(POINT_COLOR, POINT_WIDTH, POINT_HEIGHT)
        self.point_group.add(self.point)
        self.point.new_position()
        self.score = Score(30)
        self.terminate = False

    def _score(self):
        if pygame.sprite.collide_rect(self.snake.head, self.point):
            self.score.add_score(1)
            self.point.new_position()
            self.snake.extend_body()


class GameEnviroment(Game):
    def __init__(self, *args, **kwargs):
        super(GameEnviroment, self).__init__(*args, **kwargs)

    def step(self, action):
        score_beofre = self.score.score
        state = self._get_state()
        self._do_action(action)

        super().loop()

        next_state = self._get_state()

        if score_beofre - self.score.score > 0:
            reward = 1
        elif self.terminate:
            reward = -1
        else:
            reward = 0

        return state, reward, next_state, self.terminate

    def _get_state(self):
        return pygame.surfarray.array3d(self._display_surf)

    def _do_action(self, action):
        actions = {
            0: K_UP,
            1: K_DOWN,
            2: K_LEFT,
            3: K_RIGHT
        }
        pygame.event.post(pygame.event.Event(KEYDOWN, key=actions[action]))


if __name__ == '__main__':
    game = Game(SCREEN_SIZE, FPS)
    game.execute()
