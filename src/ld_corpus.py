# coding: utf-8


"""
Utility for livedoor corpus test
```
cd <repository root>
python src/ld_fetch_data.py
```
-h option to see detail 
"""


import sys
import os
import glob
import argparse
import configparser
import subprocess
import tarfile 
import tempfile
from urllib.request import urlretrieve
import json
import pandas as pd
import re

# packages related to model
import tensorflow as tf
import modeling
import optimization
from utils import str_to_value
from run_classifier import model_fn_builder
from run_classifier import file_based_input_fn_builder
from run_classifier import file_based_convert_examples_to_features
from run_classifier import LivedoorProcessor

import tokenization_sentencepiece as tokenization
# import tokenization_sp_mod as tokenization

# evaluation utilities
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_config(path=None):
    
    if path is None:
        path = 'config.ini'
    
    config = configparser.ConfigParser()
    config.read(path)
    return config


def fetch_and_decomp_raw_data(url, cache_path, ext_dir_path):
    
    if not os.path.exists(cache_path): 
        urlretrieve(url, cache_path)
    
    if not os.path.exists(ext_dir_path):
        mode = "r:gz"
        tar = tarfile.open(cache_path, mode) 
        tar.extractall(ext_dir_path) 
        tar.close()


def extract_text_from_file_ld(filename):
    """
    text extraction rule for livedoor corpus
    """
    table = str.maketrans({
            '\n': '',
            '\t': '　',
            '\r': '',
        })
    
    with open(filename) as text_file:
        # 0: URL, 1: timestamp
        text = text_file.readlines()[2:]
        text = [sentence.strip() for sentence in text]
        text = list(filter(lambda line: line != '', text))
        return ''.join(text).translate(table)


def walk_data_directory(data_dir_path):
    """
    sample collecting rule for livedoor corpus
    """
    categories = [ 
            name for name in os.listdir(data_dir_path) 
            if os.path.isdir(os.path.join(data_dir_path, name))
        ]
    categories = sorted(categories)

    all_texts = []
    all_labels = []
    for cat in categories:
        files = glob.glob(os.path.join(data_dir_path, cat, "{}*.txt".format(cat)))
        files = sorted(files)
        
        texts = [extract_text_from_file_ld(_) for _ in files]
        labels = [cat] * len(texts)
    
        all_texts.extend(texts)
        all_labels.extend(labels)
    
    return pd.DataFrame({'text' : all_texts, 'label' : all_labels})


def cmd_fetch(args):
    """
    Entrypoint of fetch mode
    """
    config = get_config()
    local_config = config[args.config_name]
    data_root_dir = os.path.join(local_config['EXPANDED_DATA_DIR'])
    
    fetch_and_decomp_raw_data(
            url=local_config['DOWNLOAD_URL'],
            cache_path=local_config['DOWNLOADED_DATA_PATH'],
            ext_dir_path=local_config['EXPANDED_DATA_DIR']
        )
    
    df = walk_data_directory(os.path.join(data_root_dir, 'text'))
    
    # randomize whole data order
    df = df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    
    len_valid = int(len(df)*float(local_config['VALID_PROP']))
    len_test = int(len(df)*float(local_config['TEST_PROP']))
    
    df_valid = df[:len_valid]
    df_test = df[len_valid:len_valid+len_test]
    
    for prop in local_config['TRAIN_PROPS'].split(','):
        df_train = df[len_valid+len_test:].sample(frac=float(prop), random_state=args.random_state)
        
        output_dir_path = os.path.join(data_root_dir, 'prop_'+prop.replace('.', 'p'))
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        
        df_valid.to_csv(os.path.join(output_dir_path, 'test.tsv'), sep='\t', index=False)
        df_test.to_csv(os.path.join(output_dir_path, 'dev.tsv'), sep='\t', index=False)
        df_train.to_csv(os.path.join(output_dir_path, 'train.tsv'), sep='\t', index=False)


class Flags(object):
    """
    Make parameters to reconstruct estimator and tokenizer.
    """
    @staticmethod
    def get_latest_ckpt_path(dir_path):
        output_ckpts = glob.glob("{}/model.ckpt*.index".format(dir_path))
        latest_ckpt = sorted(
                output_ckpts, 
                key=lambda _: int(re.findall('model.ckpt-([0-9]+).index', _)[0])
            )[-1]
        return latest_ckpt.strip('.index')
    
    def __init__(self, args, config):
        # tokenizer settings
        self.model_file = os.path.join('model', args.sp_prefix+'.model')
        self.vocab_file = os.path.join('model', args.sp_prefix+'.vocab')
        self.do_lower_case = True
        
        # task processor
        self.task_proc = getattr(sys.modules[__name__], args.task_proc_name)()
        
        # model settings
        self.init_checkpoint = self.get_latest_ckpt_path(args.trained_model_path)
        self.max_seq_length = int(config['BERT-CONFIG']['max_position_embeddings'])
        self.use_tpu = False
        self.predict_batch_size = 4
        self.num_labels = len(self.task_proc.get_labels())
        
        # test dataset directory (not used for reconstruction)
        self.data_dir = args.test_data_dir
        
        # The following parameters are not used in predictions.
        # Just use to create RunConfig.
        self.output_dir = "/dummy"
        self.master = None
        self.save_checkpoints_steps = 1
        self.iterations_per_loop = 1
        self.num_tpu_cores = 1
        self.learning_rate = 0
        self.num_warmup_steps = 0
        self.num_train_steps = 0
        self.train_batch_size = 0
        self.eval_batch_size = 0


def get_from_list_to_examples(task_proc):
    """
    Return a function that converts 2d list (from csv) into example list
    This can be different between DataProcessors
    """
    if isinstance(task_proc, LivedoorProcessor):
        return lambda l: task_proc._create_examples(l, "test")
    else:
        raise NotImplementedError('from_list_to_examples for %s is required '%(type(FLAGS.task_proc)))
    

def load_estimator(config, FLAGS):
    
    bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
    bert_config_file.write(json.dumps({k:str_to_value(v) for k,v in config['BERT-CONFIG'].items()}))
    bert_config_file.seek(0)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file.name)
    
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host
                )
        )

    model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(FLAGS.task_proc.get_labels()),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=FLAGS.num_train_steps,
            num_warmup_steps=FLAGS.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu
        )

    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size
        )
    
    return estimator


def read_data(dataset_path, from_list_to_examples, class_labels, max_seq_length, use_tpu, tokenizer):
    """
    Read dataset file and prepare feature list acceptable to estimators
    """
    
    df_src = pd.read_csv(dataset_path, sep='\t')
    
    input_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.tf_record')
    
    file_based_convert_examples_to_features(
            from_list_to_examples([list(df_src)] + df_src.values.tolist()),
            class_labels,
            max_seq_length, 
            tokenizer,
            input_file.name
        )
    
    input_fn = file_based_input_fn_builder(
            input_file=input_file.name,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=True if use_tpu else False,
        )
    
    # Input_file object has to be kept 
    # by the caller during prediction not to be deleted
    return df_src, input_file, input_fn


def evaluate(df_src, label_list, prediction):
    
    df_src['predict'] = [label_list[elem['probabilities'].argmax()] for elem in prediction]
    
    report = []
    report.append('Accuracy:')
    report.append(sum(df_src['label'] == df_src['predict']) / len(df_src))
    report.append('Detailed report:')
    report.append(classification_report(df_src['label'], df_src['predict']))
    report.append('Confusion matrix:')
    report.append(confusion_matrix(df_src['label'], df_src['predict']))
    
    return '\n'.join([str(_) for _ in report])


def cmd_test(args):
    """
    Entrypoint of test mode
    """
    if args.trained_model_path is None:
        print('specify --trained_model_path/-p')
        sys.exit(1)
    
    if args.test_data_dir is None:
        print('specify --test_data_dir/-d')
        sys.exit(1)
    
    config = get_config()
    FLAGS = Flags(args, config)
    
    # Model
    estimator = load_estimator(config, FLAGS)
    tokenizer = tokenization.FullTokenizer(
            model_file=FLAGS.model_file,
            vocab_file=FLAGS.vocab_file,
            do_lower_case=FLAGS.do_lower_case,
        )
    
    # Test dataset
    dataset_path = os.path.join(FLAGS.data_dir, 'test.tsv')
    from_list_to_examples = get_from_list_to_examples(FLAGS.task_proc)
    df_input_src, input_file, input_fn = read_data(
            dataset_path=dataset_path,
            from_list_to_examples=from_list_to_examples, 
            class_labels=FLAGS.task_proc.get_labels(), 
            max_seq_length=FLAGS.max_seq_length, 
            use_tpu=FLAGS.use_tpu, 
            tokenizer=tokenizer,
        )
    
    prediction = estimator.predict(input_fn=input_fn)
    report_str = evaluate(df_input_src, FLAGS.task_proc.get_labels(), prediction)
    
    print('***** RESULT *****')
    print('Test dataset:')
    print(dataset_path)
    print('Tested model:')
    print(FLAGS.init_checkpoint)
    print(report_str)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='livedoor corpus test')
    parser.add_argument('--mode','-m', choices=['fetch', 'test'], required=True, help='process mode')
    parser.add_argument('--random_state','-r', type=int, default=23, help='random state')
    parser.add_argument('--config_name','-c', type=str, default='FINETUNE-LIVEDOOR-CORPUS', help='local config name')
    parser.add_argument('--sp_prefix','-s', type=str, default='wiki-ja', help='[test] sentencepiece model prefix')
    parser.add_argument('--trained_model_path','-p', type=str, default=None, help='[test] path to trained model directory')
    parser.add_argument('--test_data_dir','-d', type=str, default=None, help='[test] path to a directory contained test.tsv')
    parser.add_argument('--task_proc_name','-t', type=str, default='LivedoorProcessor', help='[test] Task descriptor.')
    args = parser.parse_args()
    
    getattr(sys.modules[__name__], 'cmd_'+args.mode)(args)

