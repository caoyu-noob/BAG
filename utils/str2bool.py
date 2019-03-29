import argparse

def str2bool(text):
    if text.lower() in ('true', 'yes', 'y'):
        return True
    elif text.lower() in ('false', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')