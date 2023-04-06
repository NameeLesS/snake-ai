import pygame
import random

from config import *


class Point(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        pygame.draw.rect(
            self.image,
            color,
            pygame.Rect(0, 0, width, height)
        )
        self.rect = self.image.get_rect()

    def new_position(self):
        position = pygame.Vector2(
            random.randint(0, CELL_NUMBER - 1),
            random.randint(0, CELL_NUMBER - 1),
        )
        new_position = position * CELL_SIZE
        self.rect.center = new_position
