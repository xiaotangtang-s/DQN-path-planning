import cv2
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
# import tkinter as tk
import time
# from grid_env_2 import BuildMaze
import random as rd
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

cap = cv2.VideoCapture(0)  # 替换成视频文件路径或者0代表使用默认第一个摄像头
UNIT = 20  # 像素值

"""# 读取视频流的第一帧，并得到视频的宽度和高度
# ret 布尔型(True或者False),代表有没有读取到图片  frame 表示截取到的一帧的图片的数据，是个三维数组
ret, frame = cap.read()
# 使用cv2.threshold函数对图像进行二值化处理
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
# ret, binary_image = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)"""
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:  # 按下Esc键拍照
        cv2.imwrite('photo.jpg', frame)
        break


cap.release()
cv2.destroyAllWindows()

UNIT = 20  # 像素值

# img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

threshold_value = 128
max_value = 255
ret, threshold_image = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)

cv2.imshow('Threshold Image', threshold_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
# 使用cv2.resize函数调整图像大小，使其变为特定尺寸的栅格化图像
grid_image = cv2.resize(threshold_image, (800, 800))
height, width = grid_image.shape[:2]  # 获取grid_image的高和宽，2是python中的切片操作
# height, width = grid_image.shape[:]
# print(grid_image.shape[:2])
# 设置栅格的宽度和高度，栅格的位置将根据这些值进行计算
grid_width = 20
grid_height = 20
rows = int(height / grid_height)
cols = int(width / grid_width)
# 创建一个空的numpy数组，用于存储栅格化的图像。在这个数组中，每个单元格的值将是该单元格中像素的平均灰度值
grid_image_np = np.zeros((rows, cols), dtype=np.uint8)


# start_time = time.time()  # 获取当前时间


class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.counter = 0
        self.counter_2 = 0 # rect里的计数器
        self.origin = np.array([400, 400])
        # self.rect = np.array([0, 0]) # ------------------------------有问题-----------------------------
        # self.oval = np.array([600, 600])
        # self.title('Analog map')
        # self.geometry('{0}x{1}'.format(width, height))
        # self._build_maze()
        self.UNIT = 20  # 像素值
        self.rows = 40
        self.cols = 40
        # self.rect = 0
        # self.oval = 0
        self.cre_value = np.array([0, 0])
        self.value = np.array([20, 20])
        # self.title('Analog map')
        # self.geometry('{0}x{1}'.format(width, height))
        self._build_maze()

    def _build_maze(self):
        # 尝试截取视频流中的一帧
        # while True:
        #     self.ret, self.frame = cap.read()
        #     current_time = time.time()
        #     elapsed_time = current_time - start_time
        #
        #     if elapsed_time >= 5:
        #         break
        # 使用cv2.threshold函数对图像进行二值化处理
        # self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # ret, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        # self.ret, self.binary_image = cv2.threshold(self.gray, 80, 255, cv2.THRESH_BINARY)
        # # 使用cv2.resize函数调整图像大小，使其变为特定尺寸的栅格化图像
        # self.grid_image = cv2.resize(self.binary_image, (800, 800))
        if ret:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 原frame
            # resized = cv2.resize(gray, (cols * grid_width, rows * grid_height))
            for r in range(rows):
                for c in range(cols):
                    # cell = resized[r*grid_height:(r+1)*grid_height, c*grid_width:(c+1)*grid_width]
                    self.cell = grid_image[r * grid_height:(r + 1) * grid_height,
                                c * grid_width:(c + 1) * grid_width]
                    self.avg_gray = int(np.average(self.cell))
                    grid_image_np[r, c] = self.avg_gray
                    # grid_image[r, c] = self.avg_gray
            # cv2.imshow('grid image', grid_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # grid_final = np.resize(grid_image_np, (800, 800))
        # plt.imshow(grid_image_np, cmap='hot', interpolation='nearest')
        # plt.show()
        # 创建Tkinter窗口和Canvas对象
        # self.root = tk.Tk()
        # self.canvas = tk.Canvas(self.root, height=800, width=800)
        self.canvas = tk.Canvas(self, height=800, width=800)
        # 宽度和高度为数组的大小
        self.width = len(grid_image_np[0])
        self.height = len(grid_image_np[:, 0])
        # 计算单元格的大小
        # cell_size = min(canvas.winfo_width()/width, canvas.winfo_height()/height)
        self.cell_size = 1
        # i = 0 # 白色区域的计数器
        self.j = 0  # 黑色区域的计数器
        # safe_pos = [] #存储部分白色的点
        self.unsafe_pos = []  # 存储部分黑色的点
        # 尝试找到起点位置
        self.min_x = 40
        self.min_y = 40
        # 尝试找到终点位置
        self.max_x = 0
        self.max_y = 0
        # 逐个单元格地绘制矩形和填充文本
        for x in range(self.width):
            for y in range(self.height):
                self.cell_value = grid_image_np[x][y]

                # self.cell_x = x * self.cell_size * 20
                # self.cell_y = y * self.cell_size * 20
                # self.canvas.create_rectangle(self.cell_x, self.cell_y, self.cell_x + self.cell_size * 20,
                #                              self.cell_y + self.cell_size * 20,
                #                              fill="white")
                # self.canvas.create_text(self.cell_x + self.cell_size / 2 * 20, self.cell_y + self.cell_size / 2 * 20,
                #                         text=str(self.cell_value), fill='black')

                # if cell_value > 80 & i < 800:  # 这一步是在尝试将白色区域放在一个数组中，i大于一定的步数之后重新置0
                # safe_pos[i] = cell_value

                # 设置障碍物
                self.cell_value = grid_image_np[x][y]
                if self.cell_value < 128 and self.j < 1600:
                    arr = np.array([x * UNIT + UNIT // 2, y * UNIT + UNIT // 2])
                    self.unsafe_pos.append(
                        self.canvas.coords(self.canvas.create_rectangle(arr[0] - 10, arr[1] - 10, arr[0] + 10,
                                                                        arr[1] + 10,
                                                                        fill='black')))
                    self.j += 1
                else:
                    pass

                # elif self.cell_value < 128 and self.j >= 1600:
                #     self.j = 0
                #     arr = np.array([x * UNIT + UNIT // 2, y * UNIT + UNIT // 2])
                #     self.unsafe_pos.append(
                #         self.canvas.coords(self.canvas.create_rectangle(arr[0] - 10, arr[1] - 10, arr[0] + 10,
                #                                                         arr[1] + 10,
                #                                                         fill='black')))
                if self.cell_value > 80 and x <= self.min_x and y <= self.min_y:
                    # i += 1
                    # if x <= min_x:
                    #  min_x = x
                    # elif y <= min_y:
                    #  min_y = y
                    self.min_x = x
                    self.min_y = y
                    # print(self.min_x)
                # elif cell_value > 80 & i >= 800:
                # i = 0
                # safe_pos[i] = cell_value
                # elif self.cell_value < 80 & j < 1600:
                elif self.cell_value < 80 and x >= self.max_x and y >= self.max_y:
                    # arr = np.array([x * UNIT, y * UNIT])
                    # self.unsafe_pos[j] = self.canvas.create_rectangle(arr[0] + 1, arr[1] + 1, arr[0] + 19,
                    #                                                   arr[1] + 19,
                    #                                                   fill='black')
                    # j += 1
                    # if x <= min_x:
                    #  min_x = x
                    # elif y <= min_y:
                    #  min_y = y
                    # if x >= self.max_x & y >= self.max_y:
                    self.max_x = x
                    self.max_y = y
                # elif j >= 1600:
                #     j = 0
                # if cell_value < 80 & j < 800:  # 这一步是在尝试将黑色区域放在一个数组中，j大于一定的步数之后重新置0
                # unsafe_pos[j] = cell_value
                # j += 1
                # if cell_value < 80 :
                #     if x >= max_x:
                #         max_x = x
                #     elif y >= max_y:
                #         max_y = y
                # elif cell_value < 80 & j >= 800:
                #     j = 0
                #     # unsafe_pos[j] = cell_value
                #     if x >= max_x:
                #         max_x = x
                #     elif y >= max_y:
                #         max_y = y

        # create grids
        for c in range(0, self.width * UNIT, UNIT):  # 画每一列,中间间隔20个像素点
            x0, y0, x1, y1 = c, 0, c, self.height * UNIT
            self.canvas.create_line(x0, y0, x1, y1)  # canvas.create_line(x1,y1,x2,y2,width,fill,dash)
        for r in range(0, self.height * UNIT, UNIT):  # 画每一行，中间间隔20个像素点
            x0, y0, x1, y1 = 0, r, self.width * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.pack()

        # # # 设置起点
        # # self.origin = np.array([self.min_x * UNIT, self.min_y * UNIT])
        # # self.rect = self.canvas.create_rectangle(
        # #     self.origin[0] + 1, self.origin[1] + 1,
        # #     self.origin[0] + 19, self.origin[1] + 19,
        # #     fill='red'
        # # )
        # self.rect = self.canvas.create_rectangle(
        #     self.origin[0] - 10, self.origin[1] - 10,
        #     self.origin[1] + 10, self.origin[1] + 10,
        #     fill='red'
        # )  # 起点位于左上角
        # self.start = self.canvas.coords(self.rect)
        #
        # # 设置终点
        # # self.terminal = np.array([self.max_x * UNIT, self.max_y * UNIT])
        # # self.oval = self.canvas.create_oval(self.terminal[0] + 1, self.terminal[1] + 1,
        # #                                     self.terminal[0] + 19, self.terminal[1] + 19,
        # #                                     fill='yellow')
        # # self.oval_position = self.canvas.coords(self.oval)
        # # create oval
        # oval_center = self.origin + UNIT * 30
        # self.oval = self.canvas.create_oval(
        #     oval_center[0] - 10, oval_center[1] - 10,
        #     oval_center[0] + 10, oval_center[1] + 10,
        #     fill='yellow')
        # self.oval_position = self.canvas.coords(self.oval)
        # self.canvas.pack()

        # 设置障碍物
        # for x in range(self.width):
        #     for y in range(self.height):
        #         self.cell_value = grid_image_np[x][y]
        #         if self.cell_value < 128 & j < 1600:
        #             arr = np.array([x * UNIT, y * UNIT])
        #             self.unsafe_pos[j] = self.canvas.create_rectangle(arr[0] + 1, arr[1] + 1, arr[0] + 19,
        #                                                               arr[1] + 19,
        #                                                               fill='black')
        #             j += 1
        #         elif self.cell_value < 128 & j >= 1600:
        #             j = 0
        #             self.unsafe_pos[j] = self.canvas.create_rectangle(arr[0] + 1, arr[1] + 1, arr[0] + 19,
        #                                                               arr[1] + 19,
        #                                                               fill='black')
        print('Please set start point and final point in four seconds.')

    def on_click(self, event):

        if self.counter >= 2:
            # 在两个点之后，取消鼠标的绑定
            self.canvas.unbind("<Button-1>")
            return
        shape = self.draw_shape()

        if shape == "rect":
            # self.canvas.delete(self.rect)
            # # 计算鼠标点击位置所在的栅格位置
            # row = int((event.y - 1) / UNIT)
            # col = int((event.x - 1) / UNIT)
            # # 计算出栅格的中心位置
            # center_x = col * UNIT + UNIT / 2
            # center_y = row * UNIT + UNIT / 2
            # # 计算出偏移量
            # offset_x = center_x - event.x
            # offset_y = center_y - event.y
            # # 绘制红色矩形
            # # self.rect = self.canvas.create_rectangle(event.x-10, event.y-10, event.x+10, event.y+10, fill="red")
            # self.rect = self.canvas.create_rectangle(event.x + 1, event.y + 1, event.x + 19, event.y + 19, fill="red")
            # shape_id = self.rect
            # self.canvas.move(shape_id, -offset_x, -offset_y)
            center_x = (event.x // UNIT) * UNIT + UNIT // 2
            center_y = (event.y // UNIT) * UNIT + UNIT // 2
            radius = UNIT // 2
            self.rect = self.canvas.create_rectangle(center_x - radius, center_y - radius,
                                                     center_x + radius, center_y + radius,
                                                     fill="red")

            self.cre_value = self.canvas.coords(self.rect)[0:2]
            self.origin = self.canvas.coords(self.rect)[0:2]
            self.start = self.canvas.coords(self.rect)
        elif shape == "circle":
            # self.canvas.delete(self.oval)
            # # 计算鼠标点击位置所在的栅格位置
            # row = int((event.y - 1) / UNIT)
            # col = int((event.x - 1) / UNIT)
            # # 计算出栅格的中心位置
            # center_x = col * UNIT + UNIT / 2
            # center_y = row * UNIT + UNIT / 2
            # # 计算出偏移量
            # offset_x = center_x - event.x
            # offset_y = center_y - event.y
            # # 绘制黄色圆形
            # self.oval = self.canvas.create_oval(event.x-10, event.y-10, event.x+10, event.y+10, fill="yellow")
            # # self.oval = self.canvas.create_oval(event.x + 1, event.y + 1, event.x + 19, event.y + 19, fill="yellow")
            # shape_id = self.oval
            # self.canvas.move(shape_id, -offset_x, -offset_y)
            center_x = (event.x // UNIT) * UNIT + UNIT // 2
            center_y = (event.y // UNIT) * UNIT + UNIT // 2
            radius = UNIT // 2
            self.oval = self.canvas.create_oval(center_x - radius, center_y - radius,
                                                center_x + radius, center_y + radius,
                                                fill="yellow")
            self.oval_position = self.canvas.coords(self.oval)
        else:
            pass

        self.canvas.pack()

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
        # 将canvas放置到窗口
        # self.canvas.pack()
        # 进入Tkinter事件循环
        # self.root.mainloop()
        # self.mainloop()

    # np.set_printoptions(threshold=np.inf)
    # print(grid_image_np)
    def reset(self):
        self.update()  # 更新视图
        time.sleep(0.1)
        # counter = 0
        if self.counter_2 >= 40:
            self.canvas.delete(self.rect)
            # origin = np.array([self.min_x, self.min_y])
            self.origin = self.cre_value
            # print(self.cre_value)
            self.rect = self.canvas.create_rectangle(
                self.origin[0], self.origin[1],
                self.origin[0] + 20, self.origin[1] + 20,
                fill='red')
            self.value = (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                    self.height * 20)
            return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                self.height * 20)
        else:
            self.counter_2 += 1
            return np.array([0, 0])

        # print(self.value)
        # return observation


    def step(self, action):
        s = self.canvas.coords(self.rect)  # coords(*args) 返回画布对象的坐标(x1,y1,x2,y2) -----为什么这里的s是1*2的数组，而不是1*4
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:  # down
            if s[1] < (self.cols - 1) * self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:  # right
            if s[0] < (self.rows - 1) * self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:  # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT

        current_coords = self.canvas.coords(self.rect)
        old_distance = np.square(current_coords[0] - self.oval_position[0]) + np.square(
            current_coords[1] - self.oval_position[1])
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        # 移动红色方块，self.move(canvas_object,x,y) 将canvas_object移动到(x,y)的位置，x是距左上角的水平距离，y是距左上角的垂直距离

        next_coords = self.canvas.coords(self.rect)  # next state
        # 计算下一状态与终点的距离
        distance = np.square(next_coords[0] - self.oval_position[0]) + np.square(next_coords[1] - self.oval_position[1])

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 200
            done = True
        elif next_coords in [self.unsafe_pos[i] for i in range(0, self.j)]:
            reward = -50
            done = True
        # print([self.canvas.coords(self.hell1[i]) for i in range(0,20)])
        elif distance > old_distance:  # 添加距离的奖励，当下一步距终点的距离比上一步大时，给一个更大的负收益
            reward = -20
            done = False
        elif distance <= old_distance:  # 当下一步距终点的距离小于等于上一步时，给一个较小的负收益
            reward = -5
            done = False
        else:  # ————————————怎么让红点不要在一个地方转圈呢————————————————————
            reward = -1
            done = False
            # if reward>=-60:
            #  done = False
            # else:
            #   done = True
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / self.height * 20
        # print(s_)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()
