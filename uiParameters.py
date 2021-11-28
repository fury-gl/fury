from ui import *

panel = Panel()
panel.color = (0, 0, 0)
resetButton = Button("Reset")

Behaviours = TextUI("BEHAVIOURS", (Width-180, 120), (255, 255, 255))
UItoggle = TextUI("Press 'U' to show parameter panel", (Width-180, 120), (55, 120, 255))


Separation = TextUI("Separation: ", (Width-245, 180), (255, 255, 255))
Alignment = TextUI("Alignment: ", (Width-245, 220), (255, 255, 255))
Cohesion = TextUI("Cohesion: ", (Width-245, 260), (255, 255, 255))

SeparationValue = TextUI("separationValue: ", (Width-245, 315), (255, 255, 255))
AlignmentValue = TextUI("alignmentValue: ", (Width-245, 365), (255, 255, 255))
CohesionValue = TextUI("cohesionValue: ", (Width-245, 415), (255, 255, 255))
NumberOfBoids = TextUI("Number of Boids: ", (Width-245, 465), (255, 255, 255))
ScaleText = TextUI("Boid-Scale (radius): ", (Width-200, 520), (255, 255, 255))

toggleSeparation = ToggleButton((Width-160, 170), 20, 20, True)
toggleAlignment = ToggleButton((Width-160, 210), 20, 20, True)
toggleCohesion= ToggleButton((Width-160, 250), 20, 20, True)

separationInput = DigitInput(10, (Width-160, 300), 80, 30)
alignmentInput = DigitInput(10, (Width-160, 350), 80, 30)
cohesionInput = DigitInput(10, (Width-160, 400), 80, 30)
numberInput = DigitInput(100, (Width-160, 450), 80, 30)

sliderScale = Slider(Width-280, 550, 40, 0, 100, 180, 10, 80)

#toggleDiagonal2 = ToggleButton((Width-160, 290), 20, 20, False)
#showPoint = ToggleButton((Width-160, 340), 20, 20, True)
