from tkinter import *


class GridDrawer:
    def __init__(self):
        self.counter = 0

    def draw_grid(self):
        for i in range(21):
            x = i * 20
            canvas.create_line(x, 0, x, 400)
            canvas.create_line(0, x, 400, x)

    def on_click(self, event):

        # if self.counter >= 2:
        #     # 在两个点之后，取消鼠标的绑定
        #     canvas.unbind("<Button-1>")
        #     return
        shape = self.draw_shape()

        if shape == "rect":
            # 绘制红色矩形
            canvas.create_rectangle(event.x - 10, event.y - 10, event.x + 10, event.y + 10, fill="red")
        elif shape == "circle":
            # 绘制黄色圆形
            canvas.create_oval(event.x - 10, event.y - 10, event.x + 10, event.y + 10, fill="yellow")
        else:
            pass

        self.counter += 1

    def draw_shape(self):

        # x = event.x // 20 * 20
        # y = event.y // 20 * 20
        if self.counter == 0:
            return "rect"
        elif self.counter == 1:
            return "circle"
        else:
            return None

    # def draw_circle(self, event):
    #     x = event.x // 20 * 20
    #     y = event.y // 20 * 20
    #     self.canvas.create_oval(x + 5, y + 5, x + 15, y + 15, fill="red")
    #     print("设置完起点")
    #     # root.after(1000, root.quit)
    #     # 取消鼠标点击绑定，并显示一条消息
    #     self.canvas.unbind("<Button-1>")
    #     print("停止画栅格")


root = Tk()

drawer = GridDrawer()

canvas = Canvas(root, width=400, height=400)
canvas.pack()

drawer.draw_grid()

canvas.bind("<Button-1>", drawer.on_click)

root.mainloop()
