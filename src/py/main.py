import time
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
print(rootPath)
sys.path.append(rootPath)
sys.path.append('/data1/xdluo/Knower')
from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.model.general_models import kge_models, ea_models, et_models
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    t = time.time()
    curPath = os.path.abspath(os.path.dirname(__file__))
    model_name = 'analogy'
    kg_task = 'lp'
    if kg_task == 'ea':
        args = load_args(curPath + "/args_ea/" + model_name + r"_args_15K.json")
    elif kg_task == 'lp':
        args = load_args(curPath + "/args_kge/" + model_name + r"_args.json")
    else:
        args = load_args(curPath + "/args_et/" + model_name + r"_args.json")

    # args.training_data = "../OpenEA_dataset_v1.1/EN_DE_15K_V1/"
    # args.word_embed = 'D:/wiki-news-300d-1M.vec'
    # args.word2vec_path = 'D:/wiki-news-300d-1M.vec'
    # args.dataset_division += '/1/'
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    kgs = read_kgs_from_folder(kg_task, args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    if kg_task == 'ea':
        model = ea_models(args, kgs)
    elif kg_task == 'lp':
        model = kge_models(args, kgs)
    else:
        model = et_models(args, kgs)
    model.get_model('Analogy')
    model.run()
    model.test()
    print("Total run time = {:.3f} s.".format(time.time() - t))