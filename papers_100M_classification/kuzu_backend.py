import kuzu
import torch
from torch_geometric.loader import NeighborLoader
from multiprocessing import cpu_count

from src.model_v2 import CoraNodeClassification
from src.utils import model_params, loader_params, device, train_mask, test_mask
from src.train_test import train, test

KUZU_BM_SIZE = 6 * 1024 ** 3  # Buffer pool size of 6GB (default is 80% of memory)
torch.manual_seed(12345)

def main():
    db = kuzu.Database('data/papers100M', buffer_pool_size=KUZU_BM_SIZE)
    feature_store, graph_store = db.get_torch_geometric_remote_backend(cpu_count())

    train_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', train_mask))
    test_loader = NeighborLoader(**loader_params, data=(feature_store, graph_store), input_nodes=('paper', test_mask))

    model = CoraNodeClassification().to(device)

    mode = 'kuzu'
    train(model, train_loader, test_loader, model_params, mode)
    test_accuracy = test(model, test_loader, mode)
    print("Final test accuracy:", test_accuracy)
    # attr = graph_store.get_all_edge_attrs()[0]
    # print(graph_store._get_edge_index(attrs[0]))
    # attr = feature_store.get_all_tensor_attrs()[0]
    # print(attr)
    # attr.attr_name = 'id'
    # attr.index = [0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,
    #       12,   13,   15,   17, 1629, 1827,  690,  655, 2524,  954, 2157,  935,
    #     1924,  781,   87, 1308, 1098, 2465,  658, 1571, 1826,  549,  581, 1254,
    #     2375, 1556, 1830,  463, 1392, 2444,  692,  812,  834, 1566,  996, 1114,
    #     2009, 1678, 1408,  115, 1096,  289, 1772,  121,  565, 1484, 2287, 1557,
    #      475, 1733, 1845, 2019, 1442, 1936, 1291, 1505, 2622, 2080, 2055,  525,
    #     1446,  597,  225,  785,   79,  416,  928, 1437, 1941, 1429, 2485, 1804,
    #     2046, 2169, 1575, 1867,  534,  614,  989,  261,  529,  276, 1504, 1356,
    #      165,  866, 1641,  476, 2346, 1672,  579, 1452, 2655, 1233, 1801,  263,
    #     1020,  800, 2613, 1434, 1535,   52,  853, 2649, 2477, 1307, 1748, 2632,
    #     2020, 2251,  570, 1286, 2216, 2012, 1384, 2276, 2146, 1825, 1425,  548,
    #      639,  842, 1824,  190, 2532, 2254, 2038, 1229,  659,  970, 2364, 2137,
    #     1315,  103, 2316,  171,  674, 1213,  683, 2067, 2275, 1458, 1421, 1220,
    #     2630, 2442, 1418, 1445, 1675,  202,  443, 1153, 1643, 1652, 2575, 2182,
    #     1329, 2206, 1004,  342, 1537,  898]
    # print(len(attr.index))
    # print(feature_store._get_tensor(attr))
    # print(len(feature_store._get_tensor(attr)))

if __name__ == '__main__':
    main()