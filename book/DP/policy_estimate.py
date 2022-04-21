import numpy as np


def policy_eval(policy, env, discount_factor=1, threshold=0.00001):
    # 策略评估-计算值函数
    # 一次性计算到值函数收敛
    V = np.zeros(env.nS)
    i = 0
    print("第%d次输出各个状态值为" % i)
    print(V.reshape(5, 5))
    print("-" * 50)

    while True:
        value_delta = 0
        # 遍历各个状态
        for s in range(env.nS):
            v = 0
            # 遍历各个动作的概率
            for a, action_prob in enumerate(policy[s]):
                # 对每个动作确认下一个状态
                # prob是状态转移概率，action_prob是动作执行概率
                for prob, next_state, reward, done in env.P[s][a]:
                    # 更新当前状态值
                    v += action_prob * (reward + prob * discount_factor * V[next_state])
            # 求出状态值更新差值，确定是否收敛
            value_delta = max(value_delta, np.abs(v - V[s]))
            V[s] = v
        i += 1
        print("第%d次输出各个状态值为" % i)
        print(V.reshape(5, 5))
        print("-" * 50)

        if value_delta < threshold:
            print("第%d次后,所得结果已经收敛,运算结束" % i)
            break

    return np.array(V)
