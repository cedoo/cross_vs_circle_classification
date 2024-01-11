import pygame
import numpy as np
from nn import *

pygame.init()
pygame.font.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)

FPS = 60

WIDTH, HEIGHT = 600, 700
ROWS = COLS = 100
TOOLBAR_HEIGHT = HEIGHT - WIDTH
PIXEL_SIZE = WIDTH / (COLS * 1.2)
OFFSET = (WIDTH - PIXEL_SIZE * COLS) / 2 - PIXEL_SIZE
BG_COLOR = BLACK
DRAW_GRID_LINES = False
CURSOR_SIZE = 2

WIN = pygame.display.set_mode((WIDTH, HEIGHT))

class Button:
    def __init__(self, x, y, width, height, color, text=None, text_color=BLACK):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.text_color = text_color
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(win, self.text_color, (self.x, self.y, self.width, self.height), 5)
       
        if self.text:
            button_font = get_font(22)
            text_surface = button_font.render(self.text, 1, self.text_color)
            win.blit(text_surface, (self.x + self.width/2 - text_surface.get_width()/2, self.y + self.height/2 - text_surface.get_height()/2))
        
    def clicked(self, X, Y):
        if X > self.x and X < self.x + self.width and Y > self.y and Y < self.y + self.height:
            return True
        else:
            return False

def get_font(size):
    return pygame.font.SysFont("impact", size)

def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(win, pixel, (OFFSET + j * PIXEL_SIZE, OFFSET + i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
            if j == 0 or i == 0 or j == COLS + 1 or i == COLS + 1:
                pygame.draw.rect(win, GRAY, (OFFSET + j * PIXEL_SIZE, OFFSET + i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

def draw(win, grid, buttons):
    win.fill(BG_COLOR)
    draw_grid(WIN, grid)
    for button in buttons:
        button.draw(win)
    pygame.display.update()

def init_grid(rows, cols, color):
    grid = []
    for i in range(rows+2):
        grid.append([])
        for j in range(cols+2):
            grid[i].append(color)

    return grid

button_y = HEIGHT - TOOLBAR_HEIGHT
buttons = [
    Button(OFFSET, button_y, 150, 50, GRAY, "erase", WHITE)
]

pygame.display.set_caption("Maluj kutasiarzu")

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BLACK)
drawing_color = WHITE
clicked = False
x1 = y1 = 0
while run:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if pygame.mouse.get_pressed()[0]:
            clicked = True
        if not pygame.mouse.get_pressed()[0]:
            clicked = False
            x1, y1 = 0, 0
        if clicked:
            x, y = pygame.mouse.get_pos()
            if x > OFFSET and y > OFFSET and x < OFFSET + 1 + PIXEL_SIZE * COLS and y < OFFSET + 1 + PIXEL_SIZE * COLS:
                cords = []
                if x1:
                    c = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
                    steps = int(2 * c // PIXEL_SIZE) + 1
                    step_list = []
                    # x_step = (x - x1) // steps
                    # y_step = (y - y1) // steps
                    # print(x_step, y_step)
                    a = (y - y1) / ((x - x1)+0.0001)
                    if a < 2 and a > -2:
                        b = y1 - x1 * a
                        for i in range(steps):
                            step_list.append(x + i/steps * (x1 - x))
                        # print(f"y = {y}, x = {x}, y1 = {y1}, x1 = {x1}")
                        for i in step_list:
                            # print(i)
                            list = [i, a * i + b]
                            cords.append(list)
                    else:
                        a = (x - x1) / ((y - y1)+0.0001)
                        b = x1 - y1 * a
                        for i in range(steps):
                            step_list.append(y + i/steps * (y1 - y))
                        for i in step_list:
                            list = [a * i + b, i]
                            cords.append(list)

                    # print(x1, y1, x, y)
                    # print(cords)
                cords.append([x, y])
                for j, i in enumerate(cords):
                    if i[0] > OFFSET and i[1] > OFFSET and i[0] < OFFSET + 1 + PIXEL_SIZE * COLS and i[1] < OFFSET + 1 + PIXEL_SIZE * COLS:
                        col = (i[0] - OFFSET) // PIXEL_SIZE
                        row = (i[1] - OFFSET) // PIXEL_SIZE
                        try:
                            for k in range(int(row) - CURSOR_SIZE, int(row) + CURSOR_SIZE + 1):
                                for l in range(int(col) - CURSOR_SIZE, int(col) + CURSOR_SIZE + 1):
                                    if np.sqrt((int(row) - k) ** 2 + (int(col) - l) ** 2) <= CURSOR_SIZE:
                                        grid[k][l] = drawing_color
                        except IndexError:
                            pass

                x1, y1 = x, y
            else:
                for button in buttons:
                    if not button.clicked(x, y):
                        continue
                    drawing_color = button.color
                    if button.text == "erase":
                        drawing_color = WHITE
                        grid = init_grid(ROWS, COLS, BLACK)

            dupa = np.asarray(grid, dtype=np.uint8)
            drawing = Image.fromarray(dupa)
            single_image_new = add_transform(drawing)
            model_0.eval()
            with torch.inference_mode():
                pred = model_0(single_image_new)

            # print(f"Output logits:\n{pred}\n")
            print("---------------")
            percentages = torch.softmax(pred, dim=1)
            for i, sign in enumerate(train_data.classes):
                print(f"It's {percentages[0][i]*100:.2f}% {sign}")
            print(f"Guess: {train_data.classes[torch.argmax(percentages)]}")
            
    draw(WIN, grid, buttons)
pygame.quit()