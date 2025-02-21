import os
import subprocess

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name+=': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt +=dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt +=dot_edge.format(id(f), id(y()))

    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file = 'graph.png'):
    
    dot_graph = get_dot_graph(output, verbose)

    # 获取dot数据并保存至文件
    # os.path.expanduser('~\\123') 返回'C:\\Users\\xuguy\\123'
    # 其作用就是把 ~ 替换成user路径
    # tmp_dir可以设置在任何地方，书作者用的mac，直接apt安装的graphviz，因此把tmp_dir设定在user目录，这是不必要的
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    # Graphviz并没有安装在c盘，因此需要改造一下路径
    cd = r'cd /d D:\graphviz\bin'
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    full_cmd = cd+r'&'+cmd
    print(f'cmd: {full_cmd}')
    subprocess.run(full_cmd, shell=True)

