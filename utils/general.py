import glob
import logging
import time
from pathlib import Path

import re


def set_logging():
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO)

def write_log(results_file, msg):
    with open(results_file, 'a') as f:
        f.write(msg+'\n')  
        
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *opt, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in opt) + f'{string}' + colors['end']



def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    path = path / time_str
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return path / sep / n  # update path

def val_tensorboard(writer,global_steps, da_segment_result, ll_segment_result, detect_result,
                     total_loss, maps, t):
    writer.add_scalar('val_loss', total_loss, global_steps)
    writer.add_scalar('Driving_area_Segment_Acc', da_segment_result[0], global_steps)
    writer.add_scalar('Driving_area_Segment_IOU', da_segment_result[1], global_steps)
    writer.add_scalar('Driving_area_Segment_mIOU', da_segment_result[2], global_steps)
    writer.add_scalar('Lane_line_Segment_Acc', ll_segment_result[0], global_steps)
    writer.add_scalar('Lane_line_Segment_IOU', ll_segment_result[1], global_steps)
    writer.add_scalar('Lane_line_Segment_mIOU', ll_segment_result[2], global_steps)
    writer.add_scalar('Detect_P', detect_result[0], global_steps)
    writer.add_scalar('Detect_R', detect_result[1], global_steps)
    writer.add_scalar('Detect_mAP@0.5', detect_result[2], global_steps)
    writer.add_scalar('Detect_mAP@0.5:0.95', detect_result[3], global_steps)