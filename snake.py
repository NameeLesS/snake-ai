import pygame

from body import Body
import sys


class Snake:
    def __init__(self, color, width, height, position):
        self.color = color
        self.width, self. height = width, height
        self.snake_group = pygame.sprite.Group()
        self.head = Body(color, width, height, position, head=True)
        self.head._owner = self
        self.snake_group.add(self.head)
        self.movement_coordinates = []
        self.collides = False

    def update(self):
        self.snake_group.update(self.movement_coordinates)
        self._clear_movement_coordinates()
        self._self_bite()
        self._wall_collide()

    def on_event(self, event):
        movement = self.head.on_event(event)
        if movement:
            self.movement_coordinates.append(movement)

    def draw(self, surface):
        self.snake_group.draw(surface)

    def extend_body(self):
        last_snake = self.snake_group.sprites()[-1]
        l_position, l_direction = last_snake.position, last_snake.direction
        n_position = l_position - l_direction

        body = Body(self.color, self.width, self.height, n_position, False)
        body.direction = l_direction
        body._owner = self
        self.snake_group.add(body)

    def _clear_movement_coordinates(self):
        if len(self.snake_group.sprites()) == 1:
            self.movement_coordinates = []

    def _self_bite(self):
        for sprite in self.snake_group.sprites():
            if pygame.sprite.collide_rect(self.head, sprite) and self.head is not sprite:
                self.collides = True

    def _wall_collide(self):
        for body in self.snake_group:
            if body.collides:
                self.collides = True

    def update_coordinates(self, coordinates, body):
        last_snake = self.snake_group.sprites()[-1]
        if last_snake is body:
            for movement in self.movement_coordinates:
                if movement[0] == coordinates[0]:
                    self.movement_coordinates.remove(coordinates)
