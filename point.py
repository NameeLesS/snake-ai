import pygame
import random


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

    def update(self):
        pass

    def new_position(self):
        screen_size = pygame.display.get_window_size()
        self.rect.center = (
            random.randint(0, screen_size[0]),
            random.randint(0, screen_size[1])
        )

