# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2023/06/13 17:38:41
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import logging
import os
import time



class Logger(object):
    '''base logger'''

    def __init__(self, args):

        self.args = args
        self._create_logger()


    def _create_logger(self):
        '''
        Initialize the logging module. Concretely, initialize the
        tensorboard and logging
        '''
        
        # judge whether the folder exits
        main_path = r'./log/' + self.args.model_name + '/'
        if not os.path.exists(main_path):
            os.makedirs(main_path)

        # get the current time string
        now_str = time.strftime("%m%d%H%M%S", time.localtime())

        # Initialize tensorboard. Set the save folder.
        if self.args.log:
            os.makedirs(main_path + now_str + '/')
            file_path = now_str + '.txt'
        else:
            file_path = 'default.txt'

        # Initialize logging. Create console and file handler
        self.logger = logging.getLogger(self.args.model_name)
        self.logger.setLevel(logging.DEBUG)  # must set
        
        # create file handler
        log_path = main_path + file_path
        self.fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        self.fh.setLevel(logging.DEBUG)
        fm = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        self.fh.setFormatter(fm)
        self.logger.addHandler(self.fh)
            
        #create console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.logger.addHandler(self.ch)

        self.now_str = now_str

    
    def end_log(self):

        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)

    
    def get_logger(self):

        try:
            return self.logger
        except:
            raise ValueError("Please check your logger creater")

    
    def get_now_str(self):

        try:
            return self.now_str
        except:
            raise ValueError("An error occurs in logger")