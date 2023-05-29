import numpy as np
# import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # 禁用tensorflow 2.0的所有内容

np.random.seed(1)  # 设置随机数，随机数仅有效一次
tf.set_random_seed(1)  # 统一设置随机种子,可产生不同Session中相同的随机数.注意:同一Session中随机数还是不同的
# 随机数用于权重的设置

# tf.random.set_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.1,
            decay_step=1000,
            decay_rate=0.96,
            reward_decay=0.9,
            global_step=0,
            e_greedy=0.9,  # 0.9的概率选择贪婪策略
            replace_target_iter=3000,  # 每隔300步更新一次target_net
            memory_size=50000,
            batch_size=320,
            # e_greedy_increment=None,
            e_greedy_increment=0.05,  # 不断缩小随机的范围
            output_graph=False, # 不输出tensorboard链接
            # output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        # tf.train.exponential_decay(learning_rate, global_step, decay_step,decay_rate,staircase=False, name=None)
        self.lr = learning_rate
        # self.lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,global_step,decay_step,decay_rate,staircase=True)
        # 指数衰减法
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        # 当e_greedy_increment为None时，epsilon恒为0.9
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # numpy.zeros(shape,dtype) 这里生成一个50000*6的数组

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # 更新target_net

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录每一步的误差

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # placeholder(dtype,shape=none,name=none)
        # self.s = tf.Variable(tf.zeros([1, self.n_features]), name='s',shape=tf.TensorShape([None,self.n_features]))  # input  当前状态
        # self.s1 = np.array([None, self.n_features])
        # self.s = tf.convert_to_tensor(self.s1)
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # self.q_target = tf.Variable(tf.zeros([1, self.n_actions]), name='Q_target',shape=tf.TensorShape([None,self.n_actions]))  # for calculating loss
        # self.q_target1 = np.array([None,self.n_actions])
        # self.q_target = tf.convert_to_tensor(self.q_target1)
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):  # with是try-except-finally的简洁写法 , tf.variable_scope用来创建图层
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2,w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, 10,\
                tf.random_normal_initializer(-0.5, 0.5), tf.constant_initializer(0.1)  # config of layers
            # tf.GraphKeys.GLOBAL_VARIABLES用于将所有的variable对象加入到collection中
            # 这里是连续赋值 w_initializer是权重，b_initializer是偏差

            # tf.random_normal_initializer(mean,stddev,seed,dtype)  tensorflow的初始化器，生成一组符合标准正态分布的 tensor 对象。
            # mean:正态分布的均值，默认值 0;stddev：正态分布的标准差， 默认值 1;seed：随机种子，指定seed的值相同生成同样的数据，一个 Python 整数;
            # dtype:数据类型,只支持浮点类型

            # tf.constant_initializer(value,dtype,verify_shape) 常量初始化器

            # first layer. collections are used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)   # y=s*w1+b1

            # tf.get_variable()获取具有这些参数的现有变量或创建一个新变量
            # tf.nn.relu(features,name=None) 计算校正线性 激励函数
            # tf.matmul(a,b) 将矩阵a乘以矩阵b，得到a*b

            # second layer. collections are used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                # self.q_eval = tf.matmul(l1, w2) + b2
                l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3',[n_l2,self.n_actions],initializer=w_initializer,collections=c_names)
                b3 = tf.get_variable('b3',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_eval = tf.matmul(l2,w3)+b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  # -------这里优化器的选择可能有问题--------
        # tf.reduce_mean() 计算各个维度上元素的平均值
        # tf.squared_difference(x,y,name) 计算张量x,y对应元素的差平方，即(x-y)*(x-y)
        # tf.train.RMSPropOptimizer() 实现RMSProp算法的优化器,关键步骤,神经网络通过这里更新权重

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        # self.s_ = tf.Variable(tf.zeros([1, self.n_features]), name='s_',shape=tf.TensorShape([None,self.n_features]))  # input
        # self.s2 = np.array([None,self.n_features])
        # self.s_ = tf.convert_to_tensor(self.s2)
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                # self.q_next = tf.matmul(l1, w2) + b2
                l2 = tf.nn.relu(tf.matmul(l1,w2)+b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3',[n_l2,self.n_actions],initializer=w_initializer,collections=c_names)
                b3 = tf.get_variable('b3',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_next = tf.matmul(l2,w3)+b3

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # hasattr(object,name)用来判断object中是否有name属性

        # numpy.hstack() 用来将多个数组的同一维度的数拼接 a=[[1],[2]],b=[[3],[4]],np.hstack(a,b)==[[1,3],[2,4]]
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size   # 当运算到一定次数以后，更新记忆库
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]  # observation添加一行

        if np.random.uniform() < self.epsilon:   # np.random.uniform()默认在[0,1)均匀分布中抽取样本
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:  # 在[0,4)间随机取数
            action = np.random.randint(0, self.n_actions)
        # print(self.s)
        return action

    def learn(self):
        self.episode_reward = tf.Variable(0.0)
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:  # 当总的运行次数整除200时，更新target_net
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:  # 当记忆次数小于记忆库中的记忆量时，从记忆库中选320个————
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)  # 当记忆次数大于记忆库中的记忆量时，从新的记忆次数中选
        batch_memory = self.memory[sample_index, :]  # 从memory列表中的sample_index位置开始往后截取
        # numpy.random.choice(a,size,replace,p)
        # 从a(一维数据)中随机抽取数字，返回指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # 迭代函数
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        '''        self.episode_reward += reward
        total_reward = self.sess.run(self.episode_reward)

        '''

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon   #  关键步骤，不断提高的过程------------------------------------
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # def plot_cost(self):
    #     import matplotlib.pyplot as plt
    #     plt.figure('error-curve')
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.show()


'''        plt.figure('total-value')
        plt.plot(self.learn_step_counter, total_reward)
        plt.ylabel('total-value')
        plt.xlabel('training steps')
        plt.show()'''


'''     plt.figure('q-value')    尝试画出q_eval的曲线
        plt.plot(np.arange((len(self.cost_his)),self.q_eval))
        plt.ylabel('reward')
        plt.xlabel('training steps')
'''
