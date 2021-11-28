import pygame
from boid import Boid
from tools import Vector
import math
import random
from matrix import *
from constants import *
from uiParameters import *

pygame.init()
window = pygame.display.set_mode(size, pygame.FULLSCREEN)
clock = pygame.time.Clock()
fps = 60

scale = 40
Distance = 5
speed = 0.0005

flock = []
#number of boids
n = 50
#radius of perception of each boid

for i in range(n):
	flock.append(Boid(random.randint(20, Width-20), random.randint(20, Height-20)))

textI = "10"
reset = False
SpaceButtonPressed = False
backSpace = False
keyPressed = False
showUI = False
clicked = False
run = True
while run:
	clock.tick(fps)
	window.fill((10, 10, 15))

	n = numberInput.value
	scale = sliderScale.value

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		if event.type == pygame.MOUSEBUTTONUP:
			clicked = True
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				run = False
			if event.key == pygame.K_r:
				reset = True
			if event.key == pygame.K_SPACE:
				SpaceButtonPressed = True

			textI = pygame.key.name(event.key)
			keyPressed = True

		if event.type == pygame.KEYUP:
			if event.key == pygame.K_BACKSPACE:
				backSpace = True
			if event.key == pygame.K_u:
				showUI = not showUI

	if reset == True or resetButton.state == True:
		flock = []
		for i in range(n):
			flock.append(Boid(random.randint(20, Width-20), random.randint(20, Height-20)))
		reset = False


	for boid in flock:
		boid.toggles = {"separation": toggleSeparation.state, "alignment": toggleAlignment.state,"cohesion": toggleCohesion.state}
		boid.values = {"separation": separationInput.value/100, "alignment": alignmentInput.value/100,"cohesion": cohesionInput.value/100}
		boid.radius = scale
		boid.limits(Width, Height)
		boid.behaviour(flock)
		boid.update()
		boid.hue += speed
		boid.Draw(window, Distance, scale)



	if showUI == True:
		panel.Render(window)
		resetButton.Render(window)
		Behaviours.Render(window)
		Separation.Render(window)
		Alignment.Render(window)
		Cohesion.Render(window)
		SeparationValue.Render(window)
		AlignmentValue.Render(window)
		CohesionValue.Render(window)
		NumberOfBoids.Render(window)
		ScaleText.Render(window)
		toggleSeparation.Render(window, clicked)
		toggleAlignment.Render(window, clicked)
		toggleCohesion.Render(window, clicked)
		separationInput.Render(window, textI, backSpace, keyPressed)
		alignmentInput.Render(window, textI, backSpace, keyPressed)
		cohesionInput.Render(window, textI, backSpace, keyPressed)
		numberInput.Render(window, textI, backSpace, keyPressed)

		sliderScale.Render(window)
	else:
		UItoggle.Render(window)
	backSpace = False
	keyPressed = False
	pygame.display.flip()
	clicked = False
pygame.quit()
