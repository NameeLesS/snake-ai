import pygame

from config import *


class Score:
    def __init__(self, font_size):
        self._score = 0
        self.font = pygame.font.SysFont('Impact', font_size)

    def draw(self, surface):
        text = self.font.render(str(self._score), True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_SIZE[0] / 2, 20))
        surface.blit(text, text_rect)

    def add_score(self, value):
        self._score += value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value
