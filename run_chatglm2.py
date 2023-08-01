# -*- encoding: utf-8 -*-
'''
@File    :   run_chatglm2.py
@Time    :   2023/07/03 17:22:39
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from src.chatglm2.main import main
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from src.chatglm2.arguments import ModelArguments, DataTrainingArguments


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    main(parser)

