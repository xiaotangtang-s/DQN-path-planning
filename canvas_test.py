import tkinter as tk

# 定义要绘制的数组
values = [
  [1, 2, 1],
  [2, 3, 2],
  [1, 2, 1]
]

# 创建Tkinter窗口和Canvas对象
root = tk.Tk()
canvas = tk.Canvas(root, width=200, height=200)

# 宽度和高度为数组的大小
width = len(values[0])
height = len(values[1])

# 计算单元格的大小
cell_size = min(canvas.winfo_width()/width, canvas.winfo_height()/height)

# 逐个单元格地绘制矩形和填充文本
for y in range(height):
  for x in range(width):
    cell_value = values[y][x]
    cell_x = x * cell_size
    cell_y = y * cell_size
    canvas.create_rectangle(cell_x, cell_y, cell_x+cell_size, cell_y+cell_size, fill="white")
    canvas.create_text(cell_x+cell_size/2, cell_y+cell_size/2, text=str(cell_value), fill="black")

# 将Canvas放置到窗口
canvas.pack()

# 进入Tkinter事件循环
root.mainloop()
