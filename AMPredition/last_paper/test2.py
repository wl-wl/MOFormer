from sklearn.cluster import KMeans
import numpy as np
import math
import matplotlib.pyplot as plt
from kneed import DataGenerator, KneeLocator


sequence_index=[345, 349, 1573, 1791, 2141, 2550, 2924, 3340, 3414, 3415, 3785, 4068, 4240]
point_x=[-0.0131,  -0.5689,  -0.2547,  -0.335 ,   1.4833 , -0.9865 , -0.1862,  -0.3146,  -0.3118,
  1.6231,  -0.4509 , -0.7384 ,  1.0781]
point_y=[0.1132,  0.3192,  0.1933,  0.2714,  0.0669,  0.4176 , 0.1768 , 0.2467,  0.2317 , 0.0442,
 0.2851,  0.3377,  0.0941]


point_x=[ 1.2418,  -0.2891,   2.4679 ,  0.083  ,  1.5295 , -0.7446  , 2.2791,  -0.6623,  -0.3302,
 -0.2368 ,  0.4538,  -0.7157 ,  0.0237,   0.0962,  -0.415 ,   1.5881 ,  0.7833 , -0.787,
 -0.3575 , -0.3989 , -0.362 ,  -0.8272,  -0.1489]
point_y=[0.1077,  0.2728,  0.0576,  0.1875,  0.1014 , 0.5548,  0.0932,  0.3613 , 0.2794,  0.2111,
 0.1138,  0.3685 , 0.1896,  0.1412,  0.3572,  0.0976,  0.1134,  0.6076 , 0.2956,  0.3449,
 0.298 ,  0.9481 , 0.1941]

sorted_coords = sorted(zip(point_x, point_y))

# 拆分回两个列表
point_x, point_y = zip(*sorted_coords)


maxx=max(point_x)
index_x=point_x.index(maxx)

maxy=max(point_y)
index_y=point_y.index(maxy)

first_point=[point_x[index_x],point_y[index_x]]
second_point=[point_x[index_y],point_y[index_y]]

x1=point_x[index_x]
y1=point_y[index_x]




def angle_between_three_points(P1, P2, P3):
    """
    计算点 P2 和点 P1、P3 之间的角度

    :param P1: tuple, 第一个点的坐标 (x1, y1)
    :param P2: tuple, 中间点的坐标 (x2, y2)
    :param P3: tuple, 第二个点的坐标 (x3, y3)
    :return: float, 角度的大小（单位：度）
    """
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3

    # 向量 u 和 v
    u = (x1 - x2, y1 - y2)
    v = (x3 - x2, y3 - y2)

    # 计算点积
    dot_product = u[0] * v[0] + u[1] * v[1]

    # 计算向量的模
    magnitude_u = math.sqrt(u[0] ** 2 + u[1] ** 2)
    magnitude_v = math.sqrt(v[0] ** 2 + v[1] ** 2)

    # 计算余弦值
    cos_theta = dot_product / (magnitude_u * magnitude_v)

    # 处理浮点数精度问题，确保 cos_theta 在 [-1, 1] 之间
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # 计算角度（弧度）
    angle_radians = math.acos(cos_theta)

    # 将角度转换为度
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees
def is_point_left_of_line(P1, P2, P):
    """
    判断点 P 是否在由 P1 和 P2 确定的直线的左侧。

    :param P1: tuple, 点 P1 的坐标 (x1, y1)
    :param P2: tuple, 点 P2 的坐标 (x2, y2)
    :param P: tuple, 点 P 的坐标 (x, y)
    :return: bool, 如果点 P 在直线的左侧返回 True，否则返回 False
    """
    x1, y1 = P1
    x2, y2 = P2
    x, y = P

    # 计算向量 v1 和 v2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x - x1, y - y1)

    # 计算叉积
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    return cross_product


dis_list=[]
angle_list=[]
knee_x=[]
knee_y=[]
for i in range(len(point_x)):
 if i != 0 and i != len(point_x) - 1:
     point=[point_x[i],point_y[i]]
     dist=is_point_left_of_line([point_x[i-1],point_y[i-1]],[point_x[i+1],point_y[i+1]],point)
     angle=angle_between_three_points([point_x[i-1],point_y[i-1]],point,[point_x[i+1],point_y[i+1]])
     dis_list.append(dist)
     angle_list.append(angle)

 else:
     dis_list.append(0)
     angle_list.append(0)
print(dis_list)
print(angle_list)
for i in range(len(dis_list)):
 if dis_list[i]<0 and angle_list[i]<150:
   knee_x.append(point_x[i])
   knee_y.append(point_y[i])



plt.scatter(point_x,point_y,c=(244 / 256, 81 / 256, 96 / 256),label='Eighth Front',s=80)
plt.scatter(knee_x, knee_y, c='green', label='Knee Points', s=80)
# plt.scatter(first_point[0],first_point[1],c='orange',label='First Points',s=80)
# plt.scatter(second_point[0],second_point[1],c='orange',label='Second Points',s=80)


plt.xlim(-1.5,3)
plt.ylim(0,1)
plt.legend(fontsize=16)
plt.xlabel('D1_(mic)')

# 设置Y轴标签
plt.ylabel('D2_(hemo)')
plt.rcParams['savefig.dpi'] = 300
plt.savefig('knee_point-t-8.png',dpi=300)
plt.show()
