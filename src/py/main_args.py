import getopt
import sys
import time
import os
'''print(os.path.abspath(os.path.join(os.path.join(os.path.dirname("__file__"),os.path.pardir), os.path.pardir)))
curPath = os.path.abspath(os.path.dirname(__file__))
print(os.path.split(curPath))
rootPath = os.path.split(curPath)[0]'''
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname("__file__"),os.path.pardir), os.path.pardir)))
from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.model.general_models import kge_models, ea_models, et_models

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    # python main_args.py entity_alignment jsonFile train/test
    t = time.time()
    model_name = "empty"
    opts, args = getopt.getopt(sys.argv[1:], '-t:-m:-o:-d:', ['help', 'filename=', 'version'])
    is_train = True
    kg_task = 'ea'
    for opt_name, opt_value in opts:
        if opt_name == '-t':
            if opt_value == 'lp':
                kg_task = 'lp'
            elif opt_value == 'et':
                kg_task = 'et'
        if opt_name == '-m':
            model_name = opt_value
        if opt_name == '-o':
            if opt_value == 'test':
                is_train = opt_value
        if opt_name == '-d':
            dataset = opt_value

    curPath = os.path.abspath(os.path.dirname(__file__))
    if kg_task == 'ea':
        args = load_args(curPath + "/args_ea/" + model_name + r"_args_15K.json")
    elif kg_task == 'lp':
        args = load_args(curPath + "/args_kge/" + model_name + r"_args.json")
    else:
        args = load_args(curPath + "/args_et/" + model_name + r"_args.json")

    try:
        dataset
    except NameError:
        pass
    else:
        args.training_data = dataset
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
    model.get_model(args.embedding_module)
    if is_train:
        model.run()
        model.test()
        print("Total run time = {:.3f} s.".format(time.time() - t))
    else:
        model.test()
        print("Total run time = {:.3f} s.".format(time.time() - t))
