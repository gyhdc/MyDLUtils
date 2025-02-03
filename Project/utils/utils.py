import json
import os

import torch


class Config:
    '''
        config类，用于记录训练参数和系统配置
        将json（字典）转为config的成员变量
        读取时，传入文件路径，自动读取
        保存时，传入文件路径，自动保存
        参数：
            config：json数据，字典类型，所有参数都保存在这里
            config.key：通过成员变量的方式，获取config的key对应的value

    '''
    def __init__(self,*l_params,**params):
        '''
            传入的数据，将json转为成员变量
        '''
        if len(params) == 0 and len(l_params) == 1 and isinstance(l_params[-1], str):
            #只传入文件路径，则读取
            self.read(l_params[-1])
        else:
            self.config = params#维护一个json（字典数据）
            self.set_attr()
    def update(self,**params):
        self.config.update(params)
        self.set_attr()
    def read(self,path):
        with open(path, "r") as f:
            self.config=json.load(f)
        self.set_attr()
    def save(self,path):
        if os.path.isdir(path):
            path=os.path.join(path,"config.json")
        with open(path,mode='w') as f:
            json.dump(self.config,f)
    def __str__(self):#展示config
        res=[]
        for key,val in self.config.items():
            res.append(f"{key} : {val} ")
        return "\n".join(res)
    def __repr__(self):#展示config
        res=[]
        for key,val in self.config.items():
            res.append(f"{key} : {val} ")
        return "\n".join(res)
    def set_attr(self,):
        '''设置字典为成员变量'''
        for key, value in self.config.items():
            setattr(self, key, value)


class BestSelector:
    '''
        记录最佳指标，json转为成员变量，包括模型（路径）保存和加载
        参数：
            bestMetrics：json数据，字典类型，所有参数都保存在这里
            bestMetrics.key：通过成员变量的方式，获取bestMetrics的key对应的value
    '''
    def __init__(self, *l_params,**params):
        self.metrics_path=None
        if len(params) == 0 and len(l_params) == 1 and isinstance(l_params[-1], str):
            self.metrics_path=l_params[-1]
            self.read(l_params[-1])
        else:
            self.bestMetrics = params
            self.set_attr()

    def __getitem__(self, key):
        return self.bestMetrics[key]

    def update(self, **params):
        self.bestMetrics = params
        self.set_attr()
    def read(self,path):
        with open(path) as f:
            data=json.load(f)
        new_data={}
        for key, val in data.items():
            if key=='model':
                if not os.path.exists(val):
                    val=os.path.join(os.path.dirname(self.metrics_path),"best.pth")
                
                try:
                    model=torch.load(val)
                except:
                    model=torch.load(val, map_location=torch.device('cpu'))
                new_data[key]=model
            else:
                new_data[key]=val
        self.bestMetrics=new_data
        self.set_attr()


    def save(self, dirPath):
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        metrics = {}
        for key, val in self.bestMetrics.items():
            if key == "model":
                pthSavePath = os.path.join(dirPath, "best.pth")
                torch.save(val, pthSavePath)
                metrics["model"] = pthSavePath
            else:
                metrics[key] = val
        with open(os.path.join(dirPath, "metrics.json"), mode='w') as f:
            json.dump(metrics, f)

    def __str__(self):
        res = []
        for key, val in self.bestMetrics.items():
            if len(str(val)) > 50:
                continue
            res.append(f"{key} : {val} ")
        return ",".join(res)
    def __repr__(self):
        res = []
        for key, val in self.bestMetrics.items():
            if len(str(val)) > 50:
                continue
            res.append(f"{key} : {val} ")
        return ",".join(res)

    def set_attr(self, ):
        for key, value in self.bestMetrics.items():
            setattr(self, key, value)  # 设置字典为成员变量


class Logs:
    '''
        记录训练日志，json转为成员变量，包括模型（路径）保存和加载
        参数：
            logs：json数据，字典类型，所有参数都保存在这里
            logs.key：通过成员变量的方式，获取logs的key对应的value
    '''
    def __init__(self,*l_params,**params):

        if len(params)==0 and len(l_params)==1 and isinstance(l_params[-1],str):

            self.read(l_params[-1])
        else:
            self.logs=params
            self.set_attr()
    def get_keys(self):
        return list(self.logs.keys())
    def update(self,**params):
        self.logs.update(params)
        self.set_attr()
    def read(self,path):

        with open(path, "r") as f:
            self.logs = json.load(f)

        self.set_attr()

    def save(self,path):
        if os.path.isdir(path):
            path=os.path.join(path,"logs.json")
        with open(path,mode='w') as f:
            json.dump(self.logs,f)
    def set_attr(self,):
        for key, value in self.logs.items():
            setattr(self, key, value)#设置字典为成员变量
    def __str__(self):#展示config

        res=[]
        for key,val in self.logs.items():
            # print(key,val)
            res.append(f"{key} : {type(val)} ")

        return "\n".join(res)
    def __repr__(self):#展示config
        res=[]
        for key,val in self.logs.items():
            res.append(f"{key} : {type(val)} ")
        return "\n".join(res)
def saveProcess(saveDir,bestMod,train_log,config):
    print(saveDir)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    bestMod.save(saveDir)
    config.save(saveDir)
    train_log.save(saveDir)
def loadProcess(saveDir):
    '''
        加载训练记录
        参数：
            saveDir：模型保存文件夹的路径
        返回：
            bestMod：BestSelector对象
            config：Config对象
            train_log：Logs对象
    '''
    bestMod=BestSelector(os.path.join(saveDir,"metrics.json"))
    config=Config(os.path.join(saveDir,"config.json"))
    train_log=Logs(os.path.join(saveDir,"logs.json"))
    return bestMod,config,train_log
if __name__ == '__main__':
    logs_path=r'D:\Desktop\深度学习\4.nlp-\生物信息\Test\models\exp_3_4_4\train_log.json'
    mod=BestSelector(r"D:\Desktop\深度学习\4.nlp-\生物信息\Test\models\exp_3_4_4\metrics.json")
    config=Config(r'D:\Desktop\深度学习\4.nlp-\生物信息\Test\models\exp_3_4_4\config.json')
    print(config)
    logs=Logs(logs_path)
