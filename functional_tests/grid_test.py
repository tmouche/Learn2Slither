from app.grid import Grid
import pygame

pygame.init()
surface = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

grid = Grid(
    10, 10
)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    surface.fill("purple")
    grid.draw(surface)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()