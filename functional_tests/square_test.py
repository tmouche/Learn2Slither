from app.square import Square
import pygame

pygame.init()
surface = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

square = Square(size=50, pos=[500,250], color="red")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    surface.fill("purple")
    square.draw(surface)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()