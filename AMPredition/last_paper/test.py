
from sklearn.cluster import KMeans
import numpy as np
import math
import matplotlib.pyplot as plt
from kneed import DataGenerator, KneeLocator


point_x=[-0.4894 , 1.9584 , 1.3027 , 0.1624 , 0.229 ,  0.2802 , 0.048 ,  0.4489, -0.1058,
  0.2687 , 1.7545 , 0.6844 , 0.2594  ,1.676 ,  1.0082 , 0.3136 ,-0.4045, -0.5386,
 -0.6035 , 0.0583 , 2.1605 ,-0.1604,  1.2383 ,-0.0719 ,-0.1325 , 1.1926 , 1.4768,
  0.2998 , 0.5818 , 0.1136 , 1.5947,  0.7002, -0.0574 , 0.8526 , 1.7577 ,-0.211,
  1.835 ,  0.199,   0.6498 ,-0.461,  -0.3151]
point_y=[0.511 , 0.142 , 0.1877, 0.3072 ,0.2904, 0.2733, 0.3436, 0.2618, 0.3764, 0.286,
 0.1575, 0.2509 ,0.2903, 0.1625, 0.2217, 0.265 , 0.4463, 0.5268, 0.8178 ,0.336,
 0.1394, 0.3888 ,0.1944 ,0.3577, 0.3848 ,0.2092 ,0.1774, 0.2673 ,0.255 , 0.3211,
 0.1695 ,0.2444, 0.3438, 0.2369 ,0.1557, 0.4206, 0.1471 ,0.3063 ,0.2541, 0.4674,
 0.4425]

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


def distance_to_line(point, line_start, line_end):
 # 计算向量u和v
     x0,y0,x1,y1,x2,y2=point[0],point[1],line_start[0],line_start[1],line_end[0],line_end[1]

     numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)

     # 计算分母
     denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

     # 计算距离
     distance = numerator / denominator


     return distance

dis_list=[]
knee_x=[]
knee_y=[]
for i in range(len(point_x)):
 point=[point_x[i],point_y[i]]
 dist=distance_to_line(point,first_point,second_point)
 dis_list.append(dist)
 print(dist)
print(dis_list)
for i in range(len(dis_list)):
 if i!=0 and i!=len(point_x)-1:
  if dis_list[i]>=dis_list[i-1] and dis_list[i]>=dis_list[i+1]:
   knee_x.append(point_x[i])
   knee_y.append(point_y[i])

plt.scatter(point_x,point_y,c=(244 / 256, 81 / 256, 96 / 256),label='Eighth Front',s=80)
plt.scatter(knee_x, knee_y, c='green', label='Knee Points', s=80)
plt.scatter(first_point[0],first_point[1],c='orange',label='First Points',s=80)
plt.scatter(second_point[0],second_point[1],c='orange',label='Second Points',s=80)


plt.xlim(-1.5,3)
plt.ylim(0,1)
plt.legend(fontsize=16)
plt.xlabel('D1_(mic)')

# 设置Y轴标签
plt.ylabel('D2_(hemo)')
plt.rcParams['savefig.dpi'] = 300
plt.savefig('knee_point-t-8.png',dpi=300)
plt.show()
