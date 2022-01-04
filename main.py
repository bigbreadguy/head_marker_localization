import argparse
import train

def percentile_type(x):
    x = int(x)
    if x > 100 or x < 0:
        raise argparse.ArgumentTypeError("Integer percentiles can only be assigned")
    return x

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TestLoop",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument("--mode", default="test", type=str, dest="mode")
        self.parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
        self.parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")

        self.parser.add_argument("--split_rate", default=10, type=percentile_type, dest="split_rate")
        self.parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
        self.parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")
        self.parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

        self.parser.add_argument("--num_mark", default=2, type=int, dest="num_mark")
        self.parser.add_argument("--cuda", default="cuda:0", choices=["cuda", "cuda:0", "cuda:1"], type=str, dest="cuda")
        self.parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")

    def parse(self, args = ""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt

if __name__ == "__main__":
    args = opts().parse()
    
    mode = args.mode
    if mode == "train":
        print("Training...")
        train.train(args)
    elif mode == "test":
        print("Not defined yet!")
    else:
        print("Not defined yet!")