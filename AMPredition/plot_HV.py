import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib.ticker import MultipleLocator


class Individual(object):
    def __init__(self, dv_list: list, obj_list):
        # self.dv = np.array(dv_list)
        self.obj = np.array(obj_list)
        # self.n_dv = len(self.dv)
        self.n_obj = len(self.obj)

    # selfがotherを支配する場合 -> True
    def dominate(self, other) -> bool:
        # print('dominate函数运行..........................')
        if not isinstance(other, Individual):
            Exception("not indiv.")

        if all(s <= o for s, o in zip(self.obj, other.obj)) and \
                any(s != o for s, o in zip(self.obj, other.obj)):
            return True
        return False


def indiv_sort(population, key=-1):
    # print('indiv_sort函数运行......................')
    popsize = len(population)
    if popsize <= 1:
        return population
    # print('population', population)
    pivot = population[0]  # 查看！！！！！！！！！！
    left = []
    right = []
    for i in range(1, popsize):
        indiv = population[i]
        # print('indiv',indiv.obj, indiv.obj[key], pivot.obj[key])
        if indiv.obj[key] <= pivot.obj[key]:
            left.append(indiv)
        else:
            right.append(indiv)
    # print('left,right', left, right)
    left = indiv_sort(left, key)
    right = indiv_sort(right, key)

    center = [pivot]
    # print('center', center)
    return left + center + right


class NonDominatedSort(object):

    def __init__(self):
        pass
        # self.pop = pop

    def sort(self, population: list, return_rank=False):
        # print('非支配解sort函数运行..............................')
        popsize = len(population)

        is_dominated = np.empty((popsize, popsize), dtype=bool)
        num_dominated = np.zeros(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=bool)
        rank = np.zeros(popsize, dtype=np.int64)

        count_true = 0
        count_false = 0
        for i in range(popsize):
            for j in range(popsize):
                # if i == j:
                #     continue
                # iがjに優越されている -> True
                dom = population[j].dominate(population[i])
                if dom == True:
                    count_true += 1
                else:
                    count_false += 1
                is_dominated[i, j] = (i != j) and dom

        # iを優越する個体の数
        is_dominated.sum(axis=(1,), out=num_dominated)
        # print('num_dominated', num_dominated)
        # print('is_dominated', is_dominated)

        fronts = []
        limit = popsize
        index_list=[]
        for r in range(popsize):
            front = []
            for i in range(popsize):
                is_rank_ditermined = not (rank[i] or num_dominated[i])
                mask[i] = is_rank_ditermined
                if is_rank_ditermined:
                    rank[i] = r + 1
                    front.append(population[i])
            index_list.append(i)
            fronts.append(front)

            limit -= len(front)
            # print('front', front)
            # print('limit', limit)

            if return_rank:
                if rank.all():
                    return rank
            elif limit <= 0:
                return fronts,index_list

            # print(np.sum(mask & is_dominated))
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception("Error: reached the end of function")


class HyperVolume(object):
    def __init__(self, pareto, ref_points: list):
        self.pareto = pareto
        self.pareto_sorted = indiv_sort(self.pareto)
        self.ref_point = np.ones(pareto[0].n_obj)
        self.ref_point = np.array(ref_points)

        self.obj_dim = pareto[0].n_obj
        self.volume = 0

        self.calcpoints = []

    def set_refpoint(self, opt="minimize"):
        pareto_arr = []
        for indiv in self.pareto:
            pareto_arr.append(indiv.obj)
        pareto_arr = np.array(pareto_arr)

        # print('pareto_arr', pareto_arr)
        minmax = max
        if opt == "maximize":
            minmax = min

        for i in range(len(self.ref_point)):
            self.ref_point[i] = minmax(pareto_arr[:, i])

    def hso(self):
        # print('hso算法函数运行............................')
        pl, s = self.obj_dim_sort(self.pareto)
        s = [pl, s]
        for k in range(self.obj_dim):
            s_dash = []

    def calculate(self, obj_dim):
        pass

    def calc_2d(self):
        if len(self.ref_point) != 2:
            return NotImplemented

        vol = 0
        b_indiv = None

        for i, indiv in enumerate(self.pareto_sorted):
            if i == 0:
                x = (self.ref_point[0] - indiv.obj[0])
                y = (self.ref_point[1] - indiv.obj[1])
            else:
                x = (b_indiv.obj[0] - indiv.obj[0])
                y = (self.ref_point[1] - indiv.obj[1])

            self.calcpoints.append([x, y])
            vol += x * y
            b_indiv = indiv
            # print(f"vol:{vol:.10f}  x:{x:.5f}  y:{y:.5f}")

        self.volume = vol
        self.calcpoints = np.array(self.calcpoints)
        return vol

    def obj_dim_sort(self, dim=-1):
        pareto_arr = []
        for indiv in self.pareto:
            pareto_arr.append(indiv.obj)

        pareto_arr = np.array(pareto_arr)
        res_arr = pareto_arr[pareto_arr[:, dim].argsort(), :]
        self.pareto_sorted = res_arr

        # print('pareto_arr,res_arr', pareto_arr, )

        return res_arr, res_arr[:, dim]

"""
special_point = [0.18469619750976562, 0.24032741785049438]  # 橙色
        plt.scatter(special_point[0], special_point[1], color='#ff840e', s=65)

        point2=[-0.3647581934928894,0.44748654961586]
        plt.scatter(point2[0], point2[1], color='#124871', s=65) # 深蓝

        point3 = [1.2405954599380493, 0.15727245807647705]
        plt.scatter(point3[0], point3[1], color='#1a669c', s=65) # 浅蓝

        point4 = [0.5574136972427368,0.19537198543548584]
        plt.scatter(point4[0], point4[1], color='#62a33c', s=65) # 绿色

        point5 = [0.14042605459690094,0.29870015382766724]
        plt.scatter(point5[0], point5[1], color='#c30d4f', s=65) # 正红色
"""

def indiv_plot(population: list,vol_first, id,color=None):
    evals = []
    for indiv in (population):
        # print(indiv)
        evals.append(indiv.obj)

    evals = np.array(evals)
    print('evals[:, 0]:',evals[:, 0])
    print('evals[:, 1]:',evals[:, 1])
    # plt.legend('HV=')
    if color=='#A7AED2':
        color="#8a8a8a"
        plt.figure(figsize=(6,6))
        # g=sns.jointplot(x=evals[:, 0], y=evals[:, 1], kind="scatter", color=color,s=15)
        # g.plot_joint(plt.scatter, color=color, edgecolor="#8092C4", s=15)
        plt.scatter(evals[:, 0], evals[:, 1], color=color, edgecolor=color, s=20)

        #fontsize=12
        plt.xlim(-1.5, 4)
        plt.ylim(0, 1)
        plt.xlabel('MIC')
        plt.ylabel('TOXI')
        plt.tight_layout()
        print("HV: ", vol_first)
        # print(hypervol_best.calcpoints)
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)

        special_point = [0.24603918194770813 ,0.051100846379995346]  # 橙色
        plt.scatter(special_point[0], special_point[1], color='#ff840e', s=65)

        point2=[-0.7022276520729065, 0.27293968200683594]
        plt.scatter(point2[0], point2[1], color='#124871', s=65) # 深蓝

        point3 = [-0.45268726348876953 ,0.2839435935020447]
        plt.scatter(point3[0], point3[1], color='#1a669c', s=65) # 浅蓝

        point4 = [0.04420752823352814, 0.11668810993432999]
        plt.scatter(point4[0], point4[1], color='#62a33c', s=65) # 绿色

        point5 = [0.8634083271026611, 0.11030068248510361]
        plt.scatter(point5[0], point5[1], color='#f6f601', s=65)  # 黄色

        point6 = [-0.06238622963428497, 0.1343868374824524]
        plt.scatter(point6[0], point6[1], color='#c30d4f', s=65) # 正红色

        plt.savefig('mic_toxi.png', dpi=300, bbox_inches='tight', transparent=False)
        plt.show()
    else:
        plt.figure(figsize=(3,3))
        title_text = 'HV=' + str(np.round(vol_first, 3))
        bbox_props = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        plt.text(0.95, 0.95, title_text, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12, bbox=bbox_props)
        plt.scatter(evals[:, 0], evals[:, 1],color=color,edgecolor="#e24a30",s=20)
        plt.xlim(-1.5, 4)
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.ylim(0, 1)
        plt.xlabel('MIC')
        plt.ylabel('TOXI')
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(str(id)+'.png', dpi=300, bbox_inches='tight', transparent=False)
        # plt.show()

def data_save(pareto, vol, ref_point, fname, ext="txt"):
    pareto_arr = []
    for indiv in pareto:
        pareto_arr.append(indiv.obj)
    pareto_arr = np.array(pareto_arr)

    delimiter = " "
    if ext == "csv":
        delimiter = ","

    np.savetxt(fname + "_pareto." + ext, pareto_arr, delimiter=delimiter)
    with open(fname + "_HV." + ext, "w") as f:
        f.write("#HyperVolume\n")
        f.write(f"{vol}\n")

        f.write("#ref_point\n")
        for p in ref_point:
            f.write(f"{p}, ")
        f.write("\n")


def ref_point(input_fname):
    x_list = []
    y_list = []
    with open(input_fname, 'r') as f:
        for line in f:
            # print(line.strip().split())
            x, y = (line.strip().split(','))
            x_list.append(float(x))
            y_list.append(float(y))
    # print(x_list, '\n', y_list, '\n', len(x_list))
    x = np.array(x_list)
    y = np.array(y_list)
    x = x.astype(float)
    y = y.astype(float)
    # print(x,'\n',y)
    return [math.ceil(np.max(x)), math.ceil(np.max(y))]
    # print('x:max:', np.max(x))
    # print('x:min:', np.min(x))
    # print('y:max:', np.max(y))
    # print('y:min:', np.min(y))


def main():
    # print('主函数main开始运行.....................')
    # input_fname = "/tmp/pycharm_project_101/src/test_table.txt"  # input file name
    input_fname='/tmp/pycharm_project_AMPPrediction/c0313_mic1(mic_hemo).txt'

    output_fname = "test_result_data"  # result file name
    ext = "txt"  # outputファイルの拡張子
    # ref_points = ref_point(input_fname)
    ref_points=[2,0.5]

    # データの取得 & non-dominated-sort
    # skiprows=1 跳过一行
    ls = open(input_fname).readlines()
    newTxt = ""
    all_list=[]
    for line in ls:
        a,b=line.strip().split(',')
        newTxt = newTxt + " ".join(line.split(','))
        all_list.append([float(a),float(b)])
    # print(newTxt)
    fo = open(input_fname+'_copy', 'w')
    fo.write(newTxt)
    fo.close()

    dat = np.loadtxt(input_fname+'_copy')
    sortfunc = NonDominatedSort()
    population = []
    # [1:6],[6:]
    for s in dat:
        population.append(Individual([], s))
        # print(population[-1].__dict__)

    front,index_list = sortfunc.sort(population)
    print('主函数回调结果front', front)

    pareto_first = front[0]  # パレート解のリスト
    pareto_last=front[-1]
    num_font = 0
    for i in range(len(front)):
        num_font += len(front[0])
    # print('num_front', num_font)

    print("Number of pareto solutions: ", len(pareto_first))
    # print('前沿上的序列索引',index_list)
    # print('front',front[0][0].obj)
    # print(front[0][0],front[0][0][0])
    # print(all_list)
    front_index_list=[]
    for i in range(len(front[0])):
        front_index_list.append(all_list.index([front[0][i].obj[0],front[0][i].obj[1]]))
    print('front_index_list（第0前沿）',front_index_list)

    front_index_list1 = []
    for i in range(len(front[1])):
        front_index_list1.append(all_list.index([front[1][i].obj[0], front[1][i].obj[1]]))
    print('front_index_list1（第1前沿）', front_index_list1)

    front_index_list2 = []
    for i in range(len(front[2])):
        front_index_list2.append(all_list.index([front[2][i].obj[0], front[2][i].obj[1]]))
    print('front_index_list2（第2前沿）', front_index_list2)

    front_index_list3 = []
    for i in range(len(front[3])):
        front_index_list3.append(all_list.index([front[3][i].obj[0], front[3][i].obj[1]]))
    print('front_index_list6（第3前沿）', front_index_list3)

    front_index_list4 = []
    for i in range(len(front[4])):
        front_index_list4.append(all_list.index([front[4][i].obj[0], front[4][i].obj[1]]))
    print('front_index_list4（第4前沿）', front_index_list4)

    front_index_list5 = []
    for i in range(len(front[5])):
        front_index_list5.append(all_list.index([front[5][i].obj[0], front[5][i].obj[1]]))
    print('front_index_list5（第5前沿）', front_index_list5)

    front_index_list6 = []
    for i in range(len(front[6])):
        front_index_list6.append(all_list.index([front[6][i].obj[0], front[6][i].obj[1]]))
    print('front_index_list6（第6前沿）', front_index_list6)

    front_index_list7 = []
    for i in range(len(front[7])):
        front_index_list7.append(all_list.index([front[7][i].obj[0], front[7][i].obj[1]]))
    print('front_index_list7（第7前沿）', front_index_list7)

    front_index_list8 = []
    for i in range(len(front[8])):
        front_index_list8.append(all_list.index([front[8][i].obj[0], front[8][i].obj[1]]))
    print('front_index_list8（第8前沿）', front_index_list8)

    front_index_list9 = []
    for i in range(len(front[9])):
        front_index_list9.append(all_list.index([front[9][i].obj[0], front[9][i].obj[1]]))
    print('front_index_list9（第9前沿）', front_index_list9)




    # calculate HV
    hypervol_best = HyperVolume(pareto_first, ref_points)
    hypervol_wrost=HyperVolume(pareto_last, ref_points)
    vol_first = hypervol_best.calc_2d()
    vol_last=hypervol_wrost.calc_2d()
    print("ref_point: ", hypervol_best.ref_point)
    print("HV: ", vol_first)
    # print('vol_all===================',vol_all)
    # HVなどの出力
    data_save(hypervol_best.pareto_sorted, vol_first, hypervol_best.ref_point, output_fname, ext=ext)

    # plot all indiv(blue) and pareto indiv(red)
    indiv_plot(population,vol_first,-1, color='#A7AED2')
    # #95BAA6 #395A57 绿色
    # #F5D78F #F1A93B 黄色
    # #f6b9b1 #e24a30 红色
    color_9 = '#f6b9b1'
    indiv_plot(front[0],HyperVolume(front[0], ref_points).calc_2d(), 0,color=color_9)
    indiv_plot(front[1],HyperVolume(front[1], ref_points).calc_2d(),1, color=color_9)
    indiv_plot(front[2],HyperVolume(front[2], ref_points).calc_2d(), 2,color=color_9)
    indiv_plot(front[3],HyperVolume(front[3], ref_points).calc_2d(),3, color=color_9)
    indiv_plot(front[4],HyperVolume(front[4], ref_points).calc_2d(), 4,color=color_9)
    indiv_plot(front[5],HyperVolume(front[5], ref_points).calc_2d(),5, color=color_9)
    indiv_plot(front[6],HyperVolume(front[6], ref_points).calc_2d(),6, color=color_9)
    indiv_plot(front[7],HyperVolume(front[7], ref_points).calc_2d(),7, color=color_9)
    indiv_plot(front[8],HyperVolume(front[8], ref_points).calc_2d(), 8,color=color_9)
    indiv_plot(front[9],HyperVolume(front[9], ref_points).calc_2d(),9, color=color_9)



    return vol_first,vol_last



if __name__ == "__main__":
    main()
    # input("Press <enter>")

# front = np.array([[11,4,4],
#                   [9,2,5],
#                   [5,6,7],
#                   [3,3,10]])


# front = np.array([[11,4],
#                   [9,2],
#                   [5,6],
#                   [3,3]])