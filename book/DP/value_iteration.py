import numpy as np
from lib.envs.gridworld import GridworldEnv

# 值函数计算方式与策略迭代不同

# 定义1个全局变量用来记录运算的次数
i_num = 1


def get_max_index(action_values):
    indexs = []
    policy_arr = np.zeros(len(action_values))

    action_max_value = np.max(action_values)

    for i in range(len(action_values)):
        action_value = action_values[i]

        if action_value == action_max_value:
            indexs.append(i)
            policy_arr[i] = 1
    return indexs, policy_arr


def change_policy(policys):
    action_tuple = []

    for policy in policys:
        indexs, policy_arr = get_max_index(policy)
        action_tuple.append(tuple(indexs))

    return action_tuple


def value_iteration(env, threshold=0.0001, discount_factor=1.0):
    # 使用全局变量
    global i_num

    # 计算状态行为值
    def one_step_lookahead(state, V):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                q[a] += reward + (prob * discount_factor * V[next_state])
        return q

    # 初始化V表
    V = np.zeros(env.nS)

    print("初始的各状态的最优行为值函数")
    print(V.reshape(env.shape))
    print("-" * 50)

    while True:
        # 停止条件
        value_delta = 0
        # 遍历每个状态
        for s in range(env.nS):
            # 计算当前状态的各行为值函数
            q = one_step_lookahead(s, V)
            # 找到最大行为值函数，来更新状态值函数
            best_action_value = np.max(q)
            # 值函数更新前后求差
            value_delta = max(value_delta, np.abs(best_action_value - V[s]))
            # 更新当前状态的值函数，即：将最大的行为值函数赋值给值当前状态，用以更新当前状态的值函数
            V[s] = best_action_value

        print("第%d次各状态的最优行为值函数" % i_num)
        print(V.reshape(env.shape))
        print("-" * 50)
        i_num += 1
        # 如果当前状态值函数更新前后相差小于阈值,则说明已经收敛,结束循环
        if value_delta < threshold:
            print("第%d次之后各状态的最优行为值函数已经收敛,运算结束" % (i_num - 1))
            break

    # 初始化策略
    policy = np.zeros([env.nS, env.nA])
    # 遍历各状态
    for s in range(env.nS):
        # 根据已经计算出的V,计算当前状态的各行为值函数
        q = one_step_lookahead(s, V)
        # 求出当前最大行为值函数对应的动作索引
        # 将初始策略中的对应的状态上将最大行为值函数方向置1,其余方向保持不变,仍为0
        # v1.0版更新内容,因为np.argmax(action_values)只会选取第一个最大值出现的索引,所以会丢掉其他方向的可能性,现在会输出一个状态下所有的可能性
        best_a_arr, policy_arr = get_max_index(q)
        # 将当前所有最优行为赋值给当前状态
        policy[s] = policy_arr

    print("最后求得的最优策略是")
    print(policy)
    print("=" * 50)
    return policy, V


if __name__ == '__main__':
    # 初始化迷宫环境
    env = GridworldEnv()
    # 值迭代函数
    policy, v = value_iteration(env)

    # 输出结果
    print("策略可能的方向值:")
    print(policy)
    print()

    print("策略网格形式 (0=up, 1=right, 2=down, 3=left):")
    # print(np.reshape(np.argmax(policy, axis=1), env.shape))
    # print()
    update_policy_type = change_policy(policy)
    print(np.reshape(update_policy_type, env.shape))
    print("")

    print("值函数的网格形式:")
    print(v.reshape(env.shape))
