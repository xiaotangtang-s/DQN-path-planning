# DQN-path-planning
Real time image capture+DQN path planning
尝试在B站莫烦大佬的代码基础上进行了修改。
加入了一些简单的opencv功能，可以通过摄像头捕捉到实时的画面，然后进行栅格化。采用栅格法进行路径规划。
算法参数可能还不是特别合理，需要后期改进。
运行方法：
在保证已经接入摄像头的情况下，直接运行run_this.py。在出现栅格环境后，鼠标点击设置起点和终点，时间控制在4S内。之后便可以开始训练啦。
算法的模型参数还不能保留下来，这是后面需要改进的点。
