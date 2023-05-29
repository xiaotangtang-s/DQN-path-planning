from tkinter import *

def on_canvas_click(event):
    x, y = event.x, event.y
    row = int((y - 1) / grid_size)
    col = int((x - 1) / grid_size)
    cx = col * grid_size + grid_size / 2
    cy = row * grid_size + grid_size / 2
    offset_x = cx - x
    offset_y = cy - y
    canvas.move(shape_id, -offset_x, -offset_y)

root = Tk()
canvas = Canvas(root, width=300, height=300)
grid_size = 50
num_rows = int(canvas.winfo_height() / grid_size)
num_cols = int(canvas.winfo_width() / grid_size)
rect = canvas.create_rectangle(0, 0, grid_size, grid_size, fill="red")
shape_id = rect
for row in range(num_rows):
    for col in range(num_cols):
        x1 = col * grid_size
        y1 = row * grid_size
        x2 = x1 + grid_size
        y2 = y1 + grid_size
        canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
canvas.pack()
canvas.bind("<Button-1>", on_canvas_click)
root.mainloop()
