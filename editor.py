import pygame
import numpy as np
import sys
from torchvision import datasets, transforms
import torch
from PIL import Image

pygame.init()
pygame.font.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 100, 0)
BLUE_DARK = (0, 0, 150)
BLUE_LIGHT = (100, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)

FPS = 60

WIDTH, HEIGHT = 600, 700
ROWS = COLS = 125
TOOLBAR_HEIGHT = HEIGHT - WIDTH
PIXEL_SIZE = WIDTH / (COLS * 1.2)
OFFSET = (WIDTH - PIXEL_SIZE * COLS) / 2 - PIXEL_SIZE
BG_COLOR = BLACK
DRAW_GRID_LINES = False
CURSOR_SIZE = 2

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
# model_0.load_state_dict(torch.load("model_hid40_batch128_acc92/model.pt"))
model_0 = torch.jit.load("model_hid82_batch192_acc86/model_scripted.pt")
model_0.to("cpu")

class Button:
    def __init__(self, x, y, width, height, color, text=None, text_color=WHITE, frame_color=WHITE, name=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.text_color = text_color
        self.frame_color = frame_color
        self.name = name
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(win, self.frame_color, (self.x, self.y, self.width, self.height), 3)
       
        if self.text:
            button_font = get_font(22)
            if self.name == "size":
                self.text = str(CURSOR_SIZE)
                button_font = get_font(20)
            if self.text == "-" or self.text == "+":
                button_font = get_font(16)
            if self.name =="info":
                button_font = get_font(20)
                self.text = save_path + "/" + str(count) + ".bmp"
            text_surface = button_font.render(self.text, 1, self.text_color)
            win.blit(text_surface, (self.x + self.width/2 - text_surface.get_width()/2, self.y + self.height/2 - text_surface.get_height()/2))
        
    def clicked(self, X, Y):
        if X > self.x and X < self.x + self.width and Y > self.y and Y < self.y + self.height:
            return True
        else:
            return False

add_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

def get_font(size):
    return pygame.font.SysFont("consolas", size)

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

button_y = HEIGHT - TOOLBAR_HEIGHT + 50
if len(sys.argv) > 1:
    save_path = "flags/" + sys.argv[1]
else:
    save_path = "dupa"
count = 1
buttons = [
    Button(OFFSET, button_y, 100, 40, GRAY, "CLEAR", WHITE),
    Button(OFFSET + 110, button_y, 100, 40, GRAY, "SAVE", WHITE),
    Button(OFFSET, button_y - 50, 40, 40, WHITE, name="colors"),
    Button(OFFSET + 50, button_y - 50, 40, 40, BLACK, name="colors"),
    Button(OFFSET + 100, button_y - 50, 40, 40, RED, name="colors"),
    Button(OFFSET + 150, button_y - 50, 40, 40, YELLOW, name="colors"),
    Button(OFFSET + 200, button_y - 50, 40, 40, GREEN, name="colors"),
    Button(OFFSET + 250, button_y - 50, 40, 40, BLUE_LIGHT, name="colors"),
    Button(OFFSET + 300, button_y - 50, 40, 40, BLUE_DARK, name="colors"),
    Button(OFFSET + 350, button_y - 50, 20, 40, BLACK, "-", WHITE, BLACK),
    Button(OFFSET + 390, button_y - 50, 20, 40, BLACK, "+", WHITE, BLACK),
    Button(WIDTH - OFFSET - 60, button_y - 50, 60, 40, GRAY, "FILL", WHITE, frame_color=WHITE),
    Button(OFFSET + 370, button_y - 50, 20, 40, BLACK, str(CURSOR_SIZE), WHITE, BLACK, "size"),
    Button(OFFSET + 270, button_y, 350, 40, BLACK, save_path + "/" + str(count) + ".bmp", WHITE, BLACK, "info")
]
pygame.display.set_caption("Maluj")

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, WHITE)
drawing_color = BLACK
clicked = False
x1 = y1 = 0
guessing = False
train_dir = "augmented_output/train/"
train_data = datasets.ImageFolder(root=train_dir, transform=add_transform)
# print(save_path)
while run:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if pygame.mouse.get_pressed()[0]:
            clicked = True
        if not pygame.mouse.get_pressed()[0]:
            clicked = False
            done = False
            x1, y1 = 0, 0
        if clicked:
            x, y = pygame.mouse.get_pos()
            if x > 0 and y > 0 and x < WIDTH and y < WIDTH:
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
                                        if k > 0 and l > 0:
                                            grid[k][l] = drawing_color
                        except IndexError:
                            pass

                x1, y1 = x, y
            else:
                if not done:
                    done = True
                    for button in buttons:
                        if not button.clicked(x, y):
                            continue
                        if button.name == "colors":
                            drawing_color = button.color
                        if button.text == "CLEAR":
                            grid = init_grid(ROWS, COLS, WHITE)
                        if button.text == "FILL":
                            grid = init_grid(ROWS, COLS, drawing_color)
                        if button.text == "SAVE":
                            drawing.save(save_path + "/" + str(count) + ".bmp")
                            count += 1
                            grid = init_grid(ROWS, COLS, WHITE)
                        if button.text == "-":
                            if CURSOR_SIZE > 1:
                                CURSOR_SIZE -= 1
                        if button.text == "+":
                            if CURSOR_SIZE < 10:
                                CURSOR_SIZE += 1

            # CONTINUOUS GUESSING CROSS VS CIRCLE
            dupa = np.asarray(grid, dtype=np.uint8)
            drawing = Image.fromarray(dupa)
            single_image_new = add_transform(drawing).unsqueeze(dim=0)
            model_0.eval()
            with torch.inference_mode():
                # print(f"trainig tensor: {single_image_new}")
                # print(f"shape: {single_image_new.shape}")
                pred = model_0(single_image_new)

            # print(f"Output logits:\n{pred}\n")
            print("---------------\nGuess:")
            percentages = torch.softmax(pred, dim=1)
            guesses = np.column_stack((train_data.classes, percentages.tolist()[0]))
            # print(percentages.argsort())
            guesses = guesses[percentages.argsort()[0]]
            # for i, sign in enumerate(train_data.classes):
        #         print(f"It's {percentages[0][i]*100:.2f}% {sign}")
            for i in range (0,10):
                if float(guesses[len(guesses) - i - 1][1]) > 0.001:
                    print(f"{guesses[len(guesses) - i - 1][0]} for {float(guesses[len(guesses) - i - 1][1])*100:.2f}%")
                # print(f"{train_data.classes[torch.argmax(percentages)]} for {percentages[0][torch.argmax(percentages)]*100:.2f}%")
            
    draw(WIN, grid, buttons)
pygame.quit()