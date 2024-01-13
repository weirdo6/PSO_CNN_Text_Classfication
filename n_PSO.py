import numpy as np
import random
import n_model as md
import tensorflow as tf


def fit_fun(param, X):  # 适应函数,此处为模型训练
    # 设置GPU按需使用
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        #tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    # 获取模型参数
    label_count = param['label_count']
    a_shape = param['a_shape']
    b_shape = param['b_shape']
    train_data = param['data']
    train_label = param['label']
    model = md.cnn_model(label_count, data_shape=(a_shape, b_shape))
    # 传入待优化参数learning_rate
    res_model = model.model_create(X[-1])
    history = res_model.fit(train_data, train_label, epochs=5, batch_size=8, validation_split=0.2)
    # 获取最小的loss值,优化为loss最小时learning_rate
    val_loss = 1 - max(history.history['val_acc'])
    return val_loss


class Particle:
    # 初始化
    def __init__(self, model_param, x_max, x_min, max_vel, dim):
        self.__pos = [random.uniform(x_min, x_max) for i in range(dim)]  # 粒子的位置
        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(model_param, self.__pos)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, model_param, pso_param, best_fitness_value=float('Inf'), C1=2,
                 C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = pso_param['dim']  # 粒子的维度
        self.size = pso_param['size']  # 粒子个数
        self.iter_num = pso_param['iter_num']  # 迭代次数
        self.x_max = pso_param['x_max']  # 粒子最大位置
        self.x_min = pso_param['x_min']  # 粒子最小位置
        self.max_vel = pso_param['max_vel']  # 粒子最大速度
        self.best_position = [0.0 for i in range(pso_param['dim'])]  # 种群最优位置
        self.model_param = model_param  # 模型参数
        self.best_fitness_value = best_fitness_value
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.model_param, self.x_max, self.x_min, self.max_vel, self.dim) for i in
                              range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        for i in range(self.dim):
            # 计算移动速度，惯性向量+自身移动向量+种群向量
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (
                    part.get_best_pos()[i] - part.get_pos()[i]) + self.C2 * random.random() * (
                                self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            # 当前位置 = 上次位置+移动距离
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        # 放入适应函数中求解
        value = fit_fun(self.model_param, part.get_pos())
        # 求解值小于初始化适应值，更新
        if part.get_fitness_value() > value >= 0 and min(part.get_pos()) >= self.x_min:
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        # 求解值小于局部最优适应值，更新
        if self.get_bestFitnessValue() > value >= 0 and min(part.get_pos()) >= self.x_min:
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())
            # 每次迭代完把当前的最优适应度存到列表
        return self.fitness_val_list[-1], self.get_bestPosition()[0]
