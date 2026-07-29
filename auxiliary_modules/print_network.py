import hiddenlayer as h
import torch

def print_a_network(model, save_path, input_size):
    vis_graph = h.build_graph(model, torch.ones(input_size))
    vis_graph.theme = h.graph.THEMES["blue"].copy()
    vis_graph.save(save_path)