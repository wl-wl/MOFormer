import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


class Individual(object):
    def __init__(self, dv_list: list, obj_list):
        # self.dv = np.array(dv_list)
        self.obj = np.array(obj_list)
        # self.n_dv = len(self.dv)
        self.n_obj = len(self.obj)


    def dominate(self, other) -> bool:
        if not isinstance(other, Individual):
            Exception("not indiv.")

        if all(s <= o for s, o in zip(self.obj, other.obj)) and \
                any(s != o for s, o in zip(self.obj, other.obj)):
            return True
        return False


def indiv_sort(population, key=-1):
    popsize = len(population)
    if popsize <= 1:
        return population
    # print('population', population)
    pivot = population[0]
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
                dom = population[j].dominate(population[i])
                if dom == True:
                    count_true += 1
                else:
                    count_false += 1
                is_dominated[i, j] = (i != j) and dom

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


def indiv_plot(population: list, color=None):
    evals = []
    for indiv in (population):
        # print(indiv)
        evals.append(indiv.obj)

    evals = np.array(evals)
    print('evals[:, 0]:',evals[:, 0])
    print('evals[:, 1]:',evals[:, 1])
    # plt.legend('HV=')
    if color=='#A7AED2':
        sns.jointplot(x=evals[:, 0], y=evals[:, 1], kind="scatter", color=color,s=15)
    else:
        plt.scatter(evals[:, 0], evals[:, 1],color=color,s=15)

def indiv_plot2(population: list, color=None):
    evals = []
    for indiv in (population):
        # print(indiv)
        evals.append(indiv.obj)

    evals = np.array(evals)
    print('evals[:, 0]:',evals[:, 0])
    print('evals[:, 1]:',evals[:, 1])
    # plt.legend('HV=')

    ax1 = plt.gca()
    ax_divider = make_axes_locatable(ax1)
    ax_top = ax_divider.append_axes("top", size=0.2, pad=0.1, sharex=ax1)
    ax_right = ax_divider.append_axes("right", size=0.2, pad=0.1, sharey=ax1)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.get_xaxis().set_visible(False)
    ax_top.axis('off')
    ax_top.set_xticks([])

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.get_yaxis().set_visible(False)
    ax_right.axis('off')


    ax_right.set_yticks([])

    sns.kdeplot(evals[:, 0], ax=ax_top, color='#999A9E', fill=True,shade_lowest=False)
    sns.kdeplot(evals[:, 1], ax=ax_right, color='#999A9E', fill=True, vertical=True,shade_lowest=False)

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
    # input_fname = "/tmp/pycharm_project_101/src/test_table.txt"  # input file name
    input_fname='/compare_toxi_result/t_proposed_prediction_copy.txt'

    output_fname = "test_result_data"  # result file name
    ext = "txt"
    ref_points=[2,0.5]


    ls = open(input_fname).readlines()
    newTxt = ""
    all_list=[]
    for line in ls:
        a,b=line.strip().split(',')
        newTxt = newTxt + " ".join(line.split(','))
        all_list.append([float(a),float(b)])

    fo = open(input_fname+'_copy', 'w')
    fo.write(newTxt)
    fo.close()

    dat = np.loadtxt(input_fname+'_copy')
    sortfunc = NonDominatedSort()
    population = []
    # [1:6],[6:]
    for s in dat:
        population.append(Individual([], s))

    front,index_list = sortfunc.sort(population)

    pareto_first = front[0]
    pareto_last=front[-1]
    num_font = 0
    for i in range(len(front)):
        num_font += len(front[0])

    print("Number of pareto solutions: ", len(pareto_first))
    front_index_list=[]
    for i in range(len(front[0])):
        front_index_list.append(all_list.index([front[0][i].obj[0],front[0][i].obj[1]]))

    front_index_list1 = []
    for i in range(len(front[1])):
        front_index_list1.append(all_list.index([front[1][i].obj[0], front[1][i].obj[1]]))

    front_index_list2 = []
    for i in range(len(front[2])):
        front_index_list2.append(all_list.index([front[2][i].obj[0], front[2][i].obj[1]]))

    front_index_list3 = []
    for i in range(len(front[3])):
        front_index_list3.append(all_list.index([front[3][i].obj[0], front[3][i].obj[1]]))


    front_index_list4 = []
    for i in range(len(front[4])):
        front_index_list4.append(all_list.index([front[4][i].obj[0], front[4][i].obj[1]]))

    front_index_list5 = []
    for i in range(len(front[5])):
        front_index_list5.append(all_list.index([front[5][i].obj[0], front[5][i].obj[1]]))

    front_index_list6 = []
    for i in range(len(front[6])):
        front_index_list6.append(all_list.index([front[6][i].obj[0], front[6][i].obj[1]]))

    front_index_list7 = []
    for i in range(len(front[7])):
        front_index_list7.append(all_list.index([front[7][i].obj[0], front[7][i].obj[1]]))

    front_index_list8 = []
    for i in range(len(front[8])):
        front_index_list8.append(all_list.index([front[8][i].obj[0], front[8][i].obj[1]]))

    front_index_list9 = []
    for i in range(len(front[9])):
        front_index_list9.append(all_list.index([front[9][i].obj[0], front[9][i].obj[1]]))

    # calculate HV
    hypervol_best = HyperVolume(pareto_first, ref_points)
    hypervol_wrost=HyperVolume(pareto_last, ref_points)
    vol_first = hypervol_best.calc_2d()
    vol_last=hypervol_wrost.calc_2d()
    print("ref_point: ", hypervol_best.ref_point)
    print("HV: ", vol_first)

    data_save(hypervol_best.pareto_sorted, vol_first, hypervol_best.ref_point, output_fname, ext=ext)

    # plot all indiv(blue) and pareto indiv(red)
    indiv_plot(population, color='#A7AED2')

    indiv_plot(front[0], color="#E99C93")
    color_9='#E99C93'
    indiv_plot(front[1], color=color_9)
    indiv_plot(front[2], color=color_9)
    indiv_plot(front[3], color=color_9)
    indiv_plot(front[4], color=color_9)
    indiv_plot(front[5], color=color_9)
    indiv_plot(front[6], color=color_9)
    indiv_plot(front[7], color=color_9)
    indiv_plot(front[8], color=color_9)
    indiv_plot(front[9], color=color_9)


    plt.xlim(-1.5,4)
    plt.ylim(0,1)
    plt.xlabel('D1_(mic)')
    # 设置Y轴标签
    plt.ylabel('D2_(hemo)')
    # plt.scatter(hypervol.calcpoints[:,0], hypervol.calcpoints[:,1], "*")
    print("HV: ",vol_first)
    print(hypervol_best.calcpoints)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.title('HV='+str(np.round(vol_first,4)))

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)

    title_text='HV='+str(np.round(vol_first,4))
    bbox_props = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    # plt.text(0.95, 0.95, title_text, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12, bbox=bbox_props)
    plt.text(0.95, 0.95, title_text, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12, bbox=bbox_props)

    plt.show()
    return vol_first,vol_last



if __name__ == "__main__":
    main()

