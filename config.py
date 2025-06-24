import configparser


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Hyper-parameter
        self.seed = conf.getint("Model_Setup", "seed")
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.batch_size = conf.getint("Model_Setup", "batch_size")
        self.warmup_updates = conf.getint("Model_Setup", "warmup_updates")
        self.tot_updates = conf.getint("Model_Setup", "tot_updates")
        self.peak_lr = conf.getfloat("Model_Setup", "peak_lr")
        self.end_lr = conf.getfloat("Model_Setup", "end_lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.patience = conf.getint("Model_Setup", "patience")
        self.hidden_dim = conf.getint("Model_Setup", "hidden_dim")
        self.laplace_dim = conf.getint("Model_Setup", "laplace_dim")
        self.n_layers = conf.getint("Model_Setup", "n_layers")
        self.num_heads = conf.getint("Model_Setup", "num_heads")
        self.dropout_rate = conf.getfloat("Model_Setup", "dropout_rate")
        self.attention_dropout_rate = conf.getfloat("Model_Setup", "attention_dropout_rate")


        # Dataset
        self.name = conf.get("Data_Setting", "name")
        self.sample_hop = conf.getint("Data_Setting", "sample_hop")
        self.sample_num = conf.getint("Data_Setting", "sample_num")









