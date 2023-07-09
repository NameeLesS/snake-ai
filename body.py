import pygame
from pygame.locals import *
from config import *

import sys


class Body(pygame.sprite.Sprite):
    def __init__(self, color, head_color, width, height, position, head=False):
        super().__init__()
        self._owner = None
        self.head = head
        self.direction = pygame.Vector2(1, 0)
        self.direction_v = 'left'
        self.position = position
        self.collides = False

        if self.head:
            color = head_color

        self.image = pygame.Surface([width, height])
        # Body
        pygame.draw.rect(self.image,
                         color,
                         pygame.Rect(0, 0, width, height))
        # Border
        # pygame.draw.rect(self.image,
        #                  (0, 0, 255),
        #                  pygame.Rect(0, 0, width, height), 1)
        self.rect = self.image.get_rect()
        self.rect.center = self.position * CELL_SIZE

    def update(self, movement_coordinates):
        self._change_direction(movement_coordinates)
        self._move()
        if self._wall_collide():
            self.collides = True

    def on_event(self, event):
        if self.head:
            return self._handle_movement(event)

    def _wall_collide(self):
        if self.rect.bottom > SCREEN_SIZE[1]:
            return True
        elif self.rect.top < 0:
            return True
        elif self.rect.left < 0:
            return True
        elif self.rect.right > SCREEN_SIZE[0]:
            return True
        else:
            return False

    def _handle_movement(self, event):
        if event.type == KEYDOWN:
            if event.key == K_UP and self.direction_v != 'down':
                return self.rect.center, pygame.Vector2(0, -1), 'up'
            if event.key == K_DOWN and self.direction_v != 'up':
                return self.rect.center, pygame.Vector2(0, 1), 'down'
            if event.key == K_RIGHT and self.direction_v != 'left':
                return self.rect.center, pygame.Vector2(1, 0), 'right'
            if event.key == K_LEFT and self.direction_v != 'right':
                return self.rect.center, pygame.Vector2(-1, 0), 'left'
        return None

    def _move(self):
        self.position += self.direction
        self.rect.center = self.position * CELL_SIZE

    def _change_direction(self, movement_coordinates):
        for movement in movement_coordinates:
            if self.rect.center == movement[0]:
                self.direction = movement[1]
                self.direction_v = movement[2]
                self._owner.update_coordinates(movement, self)
